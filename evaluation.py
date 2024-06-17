################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import os
import json
import sqlite3
import argparse
import tqdm
from process_sql import get_schema, Schema, get_sql

from exec_eval import eval_exec_match
from sqlglot.expressions import Intersect, Except, Union, Subquery, Expression, Count, Max, Min, Sum, Avg, Column, \
    Condition, And, Or, Identifier, Literal, Distinct, Not, Between, EQ, GT, LT, GTE, LTE, NEQ, In, Like, Is, \
    Exists, Limit, Order, Having, Group, Where, Ordered, Join, Select
from sqlglot.optimizer.scope import build_scope
from sqlglot.optimizer.qualify import qualify

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True

AGG_EXPRESSIONS = (Count, Max, Min, Sum, Avg)

WHERE_EXPR = (Not, Between, EQ, GT, LT, GTE, LTE, NEQ, In, Like, Is, Exists)

COND_EXPR = (And, Or)
SQL_EXPR = (Intersect, Union, Except)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def remove_agg(expr):
    """
    Remove top level aggregation from expression if present
    """
    # Instead of alias, use the actual expression
    expr = expr.unalias()
    # Remove outermost aggregation
    if isinstance(expr, AGG_EXPRESSIONS):
        expr = expr.this
    return expr


def eval_sel(pred, label):
    """
    Compare select statements of the two (i.e. expressions)
    Also check the same but ignore aggregations this time, i.e. it's okay if aggregations have been forgotten
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left

    cnt = 0
    cnt_wo_agg = 0
    pred_total = len(pred.expressions)
    label_total = len(label.expressions)
    label_expressions = [expr.copy().unalias() for expr in label.expressions]
    expressions_with_agg_removed = [remove_agg(expr) for expr in label.expressions]

    for expr in pred.expressions:
        if expr.unalias() in label_expressions:
            cnt += 1
            label_expressions.remove(expr.unalias())
        if remove_agg(expr) in expressions_with_agg_removed:
            cnt_wo_agg += 1
            expressions_with_agg_removed.remove(remove_agg(expr))
    return label_total, pred_total, cnt, cnt_wo_agg


def get_conditions(expression: Expression):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery) or (isinstance(expression,
                                                                                                             Condition) and not any(
            isinstance(expression, expr) for expr in COND_EXPR))

    conditions = []
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        # Do not count conjunctions
        if isinstance(node, Condition) and not any(isinstance(node, expr) for expr in COND_EXPR):
            conditions.append(node)

    return conditions


def remove_where_agg(expression: Expression):
    """
    Remove top level aggregation from expression if present
    """

    if isinstance(expression, WHERE_EXPR):
        return expression.this
    return expression


def eval_where(pred, label):
    """
        Compare conditions of the where part of the two (i.e. expressions)
        Also check the same but ignore aggregations this time, i.e. it's okay if aggregations have been forgotten
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left

    if pred.args.get("where") is None:
        pred_conds = []
    else:
        pred_conds = get_conditions(pred.args.get("where"))
    if label.args.get("where") is None:
        label_conds = []
    else:
        label_conds = get_conditions(label.args.get("where"))
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    label_wo_agg = [remove_where_agg(cond) for cond in label_conds]
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if remove_where_agg(unit) in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(remove_where_agg(unit))

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    """
    Compare columns of group statement (with stripped table prefix)
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left

    if pred.args.get("group") is None:
        pred_cols = []
    else:
        pred_cols = pred.args.get("group").expressions
    if label.args.get("group") is None:
        label_cols = []
    else:
        label_cols = label.args.get("group").expressions

    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [col.name for col in pred_cols]
    label_cols = [col.name for col in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    """
    Uses the group by columns. Only if group by is equal (including order, unlike whats done in the eval_group)
    and the having sections are IDENTICAL (other than the removed values before), return 1
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left
    pred_total = 1 if pred.args.get("group") is not None else 0
    label_total = 1 if label.args.get("group") is not None else 0
    cnt = 0
    pred_cols = [col.name for col in pred.args.get("group").expressions] if pred.args.get("group") is not None else []
    label_cols = [col.name for col in label.args.get("group").expressions] if label.args.get(
        "group") is not None else []
    if pred_total == label_total == 1 and pred_cols == label_cols and pred.args.get("having") == label.args.get(
            "having"):
        cnt = 1
    return label_total, pred_total, cnt


def eval_order(pred, label):
    """
    Only if pred and label ordering matches completely (including asc, desc) and they both have a limit or none has
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left
    pred_total = 1 if pred.args.get("order") is not None else 0
    label_total = 1 if label.args.get("order") is not None else 0
    cnt = 0
    # Replace alias with actual expression to compare easily
    pred = pred.copy()
    label = label.copy()
    pred_order = pred.args.get("order").copy() if pred.args.get("order") is not None else None
    if pred_order:
        for identifier in pred_order.find_all(Identifier):
            for expr in pred.expressions:
                if expr.alias_or_name == identifier.this:
                    identifier.parent.replace(expr.unalias())
    label_order = label.args.get("order").copy() if label.args.get("order") is not None else None
    if label_order:
        for identifier in label_order.find_all(Identifier):
            for expr in label.expressions:
                if expr.alias_or_name == identifier.this:
                    identifier.parent.replace(expr.unalias())

    if pred_total == label_total == 1 and pred_order == label_order and \
            ((pred.args.get("limit") is None and label.args.get("limit") is None) or (
                    pred.args.get("limit") is not None and label.args.get("limit") is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    """
    True if both have 'and' or none has 'and', and the same with 'ore' respectively
    """
    while any(isinstance(pred, expr) for expr in SQL_EXPR):
        pred = pred.left
    while any(isinstance(label, expr) for expr in SQL_EXPR):
        label = label.left

    pred_ao = set()
    label_ao = set()
    if pred.args.get("where") is not None:
        if get_first_no_subqueries(pred.args.get("where"), And) is not None:
            pred_ao.add('and')
        if get_first_no_subqueries(pred.args.get("where"), Or) is not None:
            pred_ao.add('or')
    if label.args.get("where") is not None:
        if get_first_no_subqueries(label.args.get("where"), And) is not None:
            label_ao.add('and')
        if get_first_no_subqueries(label.args.get("where"), Or) is not None:
            label_ao.add('or')
    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def eval_nested(pred, label):
    """
    Check against exact match if both are not none
    """
    pred_total = 1 if pred is not None else 0
    label_total = 1 if label is not None else 0
    cnt = 0
    if pred is not None and label is not None:
        cnt = Evaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def get_first_no_subqueries(
        expression: Expression, class_to_find: type
):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        if isinstance(node, class_to_find):
            return node


def get_all_no_subqueries(
        expression: Expression, class_to_find: type
):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    results = []
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        if isinstance(node, class_to_find):
            results.append(node)
    return results


def eval_IUEN(pred, label):
    """
    Take right side of tree for both queries, compare this using eval_nested
    Add scores for intersect, except, union
    """

    lt, pt, cnt = 0, 0, 0
    pred_intersect = get_first_no_subqueries(pred, Intersect)
    label_intersect = get_first_no_subqueries(label, Intersect)

    lt1, pt1, cnt1 = eval_nested(pred_intersect.right if pred_intersect is not None else None,
                                 label_intersect.right if label_intersect is not None else None)
    lt += lt1
    pt += pt1
    cnt += cnt1

    pred_except = get_first_no_subqueries(pred, Except)
    label_except = get_first_no_subqueries(label, Except)
    lt1, pt1, cnt1 = eval_nested(pred_except.right if pred_except is not None else None,
                                 label_except.right if label_except is not None else None)
    lt += lt1
    pt += pt1
    cnt += cnt1

    all_unions_pred = get_all_no_subqueries(pred, Union)
    all_unions_label = get_all_no_subqueries(label, Union)
    all_unions_pred = [union for union in all_unions_pred if union.__class__ == Union]
    all_unions_label = [union for union in all_unions_label if union.__class__ == Union]

    # Check that we have found an actual union as except and intersect are also of type Union
    lt1, pt1, cnt1 = eval_nested(all_unions_pred[0].right if len(all_unions_pred) > 0 else None,
                                 all_unions_label[0].right if len(all_unions_label) > 0 else None)
    lt += lt1
    pt += pt1
    cnt += cnt1
    return lt, pt, cnt


def dfs_left_iue(expression, prune=None):
    """
    Returns a generator object which visits all nodes in this tree in
    the DFS (Depth-first) order.

    Returns:
        The generator object.
    """
    stack = [expression]

    while stack:
        node = stack.pop()

        yield node

        if prune and prune(node):
            continue

        if any(isinstance(node, expr) for expr in SQL_EXPR):
            stack.append(node.left)
            continue

        for v in node.iter_expressions(reverse=True):
            stack.append(v)


def get_keywords(expression):
    res = set()

    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    for node in dfs_left_iue(expression, stop_traversing):
        if isinstance(node, Intersect):
            res.add('intersect')
        if isinstance(node, Except):
            res.add('except')
        if isinstance(node, Union) and node.__class__ == Union:
            res.add('union')
        if isinstance(node, Limit):
            res.add('limit')
        if isinstance(node, Order):
            res.add('order')
        if isinstance(node, Ordered):
            if node.args["desc"]:
                res.add('desc')
            else:
                res.add('asc')
        if isinstance(node, Having):
            res.add('having')
        if isinstance(node, Group):
            res.add('group')
        if isinstance(node, Where):
            res.add('where')
        # We have to exclude joins here as the original implementation does not consider or in join conditions
        if isinstance(node, Or) and not isinstance(node.parent, Join):
            res.add('or')
        if isinstance(node, Not):
            res.add('not')
        if isinstance(node, In):
            res.add('in')
        if isinstance(node, Like):
            res.add('like')
    return res


def eval_keywords(pred, label):
    """
    Check if they have the same SQL keywords (in the left side of for example intersect)
    Also, does not evaluate subqueries
    """
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_component1(sql):
    count = 0
    # Outermost intersect / union / except or plain select
    # The original implementation takes only the left query into account in this case (for god knows what reason)
    select = sql
    while any(isinstance(select, expr) for expr in SQL_EXPR):
        select = select.left
    # Only check outer components, not nested SQL (similar to original implementation)

    if select.args.get("where") is not None:
        count += 1
    if select.args.get("group") is not None:
        count += 1
    if select.args.get("order") is not None:
        count += 1
    if select.args.get("limit") is not None:
        count += 1
    # One per join in outer components, similar to original implementation
    if select.args.get("joins") is not None:
        count += len(select.args.get("joins"))
    # Add or count. From is not needed as there can only be "and" in from conditions in original implementation
    or_count = 0
    if select.args.get("where"):
        or_count += str(select.args.get("where")).count(" OR ")
    if select.args.get("having"):
        or_count += str(select.args.get("having")).count(" OR ")
    count += or_count
    # Add like count. From is not needed as there can only be "and" in from conditions in original implementation
    like_count = 0
    if select.args.get("where"):
        like_count += str(select.args.get("where")).count(" LIKE ")
    if select.args.get("having"):
        like_count += str(select.args.get("having")).count(" LIKE ")
    count += like_count
    return count


def subquery_count(expression: Expression):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    count = 0
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        if isinstance(node, Subquery):
            count += 1

    return count


def count_component2(sql):
    nested_count = 0
    # Outermost intersect / union / except or plain select
    # The original implementation takes only the left query into account in this case (for god knows what reason)
    select = sql
    while any(isinstance(select, expr) for expr in SQL_EXPR):
        select = select.left
    if select.args.get("from"):
        from_clause = select.args.get("from")
        nested_count += subquery_count(from_clause)
    if select.args.get("joins"):
        join_clause = select.args.get("joins")
        for join in join_clause:
            nested_count += subquery_count(join)
    if select.args.get("where"):
        where_clause = select.args.get("where")
        nested_count += subquery_count(where_clause)
    if select.args.get("having"):
        having_clause = select.args.get("having")
        nested_count += subquery_count(having_clause)
    return nested_count


def count_agg(expression: Expression):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    count = 0
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        if any(isinstance(node, agg) for agg in AGG_EXPRESSIONS):
            count += 1

    return count


def count_columns(expression: Expression):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery)

    count = 0
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        if isinstance(node, Column):
            count += 1

    return count


def count_conditions(expression: Expression):
    def stop_traversing(expression):
        # Stop traversing on Subqueries as sub-subqueries are not counted in the original implementation
        return expression is None or expression.is_leaf() or isinstance(expression, Subquery) or (isinstance(expression,
                                                                                                             Condition) and not any(
            isinstance(expression, expr) for expr in COND_EXPR))

    count = 0
    for node in expression.walk(
            bfs=False, prune=stop_traversing
    ):
        # Do not count conjunctions
        if isinstance(node, Condition) and not any(isinstance(node, expr) for expr in COND_EXPR):
            count += 1

    return count


def count_others(sql):
    count = 0
    # only take left query into account for intersect, except, union
    select = sql
    while any(isinstance(select, expr) for expr in SQL_EXPR):
        select = select.left
    agg_count = 0
    for expr in select.expressions:
        agg_count += count_agg(expr)
    if select.args.get("where"):
        agg_count += count_agg(select.args.get("where"))
    if select.args.get("group"):
        agg_count += count_agg(select.args.get("group"))
    if select.args.get("order"):
        agg_count += count_agg(select.args.get("order"))
    if select.args.get("having"):
        agg_count += count_agg(select.args.get("having"))
    if agg_count > 1:
        count += 1
    column_count = len(select.expressions)
    if column_count > 1:
        count += 1
    if select.args.get("where"):
        if count_conditions(select.args.get("where")) > 1:
            count += 1
    if select.args.get("group"):
        if len(select.args.get("group").expressions) > 1:
            count += 1
    return count


class Evaluator:
    """A simple evaluator"""

    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores
        for key, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total,
                                   'pred_total': pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        return res


def print_formated_s(row_name, l, element_format):
    template = "{:20} " + ' '.join([element_format] * len(l))
    print(template.format(row_name, *l))


def print_scores(scores, etype, include_turn_acc=True):
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    if include_turn_acc:
        levels.append('joint_all')
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    print_formated_s("", levels, '{:20}')
    counts = [scores[level]['count'] for level in levels]
    print_formated_s("count", counts, '{:<20d}')

    if etype in ["all", "exec"]:
        print('=====================   EXECUTION ACCURACY     =====================')
        exec_scores = [scores[level]['exec'] for level in levels]
        print_formated_s("execution", exec_scores, '{:<20.3f}')

    if etype in ["all", "match"]:
        print('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        print_formated_s("exact match", exact_scores, '{:<20.3f}')
        print('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

        print('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

        print('---------------------- PARTIAL MATCHING F1 --------------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
            print_formated_s(type_, this_scores, '{:<20.3f}')

    if include_turn_acc:
        print()
        print()
        print_formated_s("", turns, '{:20}')
        counts = [scores[turn]['count'] for turn in turns]
        print_formated_s("count", counts, "{:<20d}")

        if etype in ["all", "exec"]:
            print('=====================   TURN EXECUTION ACCURACY     =====================')
            exec_scores = [scores[turn]['exec'] for turn in turns]
            print_formated_s("execution", exec_scores, '{:<20.3f}')

        if etype in ["all", "match"]:
            print('\n====================== TURN EXACT MATCHING ACCURACY =====================')
            exact_scores = [scores[turn]['exact'] for turn in turns]
            print_formated_s("exact match", exact_scores, '{:<20.3f}')


def evaluate(gold, predict, db_dir, etype, kmaps, plug_value, keep_distinct, progress_bar_for_each_datapoint):
    with open(gold) as f:
        glist = []
        gseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = l.strip().split('\t')
                gseq_one.append(lstrip)

        # include the last session
        # this was previously ignored in the SParC evaluation script
        # which might lead to slight differences in scores
        if len(gseq_one) != 0:
            glist.append(gseq_one)

    # spider formatting indicates that there is only one "single turn"
    # do not report "turn accuracy" for SPIDER
    include_turn_acc = len(glist) > 1

    with open(predict) as f:
        plist = []
        pseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                plist.append(pseq_one)
                pseq_one = []
            else:
                pseq_one.append(l.strip().split('\t'))

        if len(pseq_one) != 0:
            plist.append(pseq_one)

    assert len(plist) == len(glist), "number of sessions must equal"

    evaluator = Evaluator()
    turns = ['turn 1', 'turn 2', 'turn 3', 'turn 4', 'turn > 4']
    levels = ['easy', 'medium', 'hard', 'extra', 'all', 'joint_all']

    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for turn in turns:
        scores[turn] = {'count': 0, 'exact': 0.}
        scores[turn]['exec'] = 0

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

    for i, (p, g) in enumerate(zip(plist, glist)):
        if (i + 1) % 10 == 0:
            print('Evaluating %dth prediction' % (i + 1))
        scores['joint_all']['count'] += 1
        turn_scores = {"exec": [], "exact": []}
        for idx, pg in enumerate(tqdm.tqdm(zip(p, g))):
            p, g = pg
            p_str = p[0]
            p_str = p_str.replace("value", "1")
            g_str, db = g
            # SQL uses single quotes, but some datasets use double quotes
            g_str = g_str.replace("\"", "'")
            p_str = p_str.replace("\"", "'")
            db_name = db
            db = os.path.join(db_dir, db, db + ".sqlite")
            schema = Schema(get_schema(db))
            g_sql = get_sql(schema, g_str)
            hardness = evaluator.eval_hardness(g_sql)
            # print("Hardness: Old {} - New {}".format(hardness, hardness_new))
            if idx > 3:
                idx = "> 4"
            else:
                idx += 1
            turn_id = "turn " + str(idx)
            scores[turn_id]['count'] += 1
            scores[hardness]['count'] += 1
            scores['all']['count'] += 1

            try:
                p_sql = get_sql(schema, p_str)
            except:
                # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
                p_sql = Select()

            if etype in ["all", "exec"]:
                exec_score = eval_exec_match(db=db, p_str=p_str, g_str=g_str, plug_value=plug_value,
                                             keep_distinct=keep_distinct,
                                             progress_bar_for_each_datapoint=progress_bar_for_each_datapoint)
                if exec_score:
                    scores[hardness]['exec'] += 1
                    scores[turn_id]['exec'] += 1
                    scores['all']['exec'] += 1
                    turn_scores['exec'].append(1)
                else:
                    turn_scores['exec'].append(0)

            if etype in ["all", "match"]:
                # rebuild sql for value evaluation
                kmap = kmaps[db_name]
                g_valid_col_units = build_valid_col_units(g_sql, schema)
                try:
                    g_sql = qualify(g_sql, schema=schema.to_sqlglot(), dialect='sqlite', quote_identifiers=False,
                                    identify=False, expand_stars=False)
                except:
                    pass
                g_sql = remove_val(g_sql)
                g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)

                p_valid_col_units_new = build_valid_col_units(p_sql, schema)
                try:
                    p_sql = qualify(p_sql, schema=schema.to_sqlglot(), dialect='sqlite', quote_identifiers=False,
                                        identify=False, expand_stars=False)
                except:
                    pass

                p_sql = remove_val(p_sql)
                p_sql = rebuild_sql_col(p_valid_col_units_new, p_sql, kmap)

                exact_score = evaluator.eval_exact_match(p_sql, g_sql)
                partial_scores = evaluator.partial_scores
                if exact_score == 0:
                    turn_scores['exact'].append(0)
                    print("{} pred: {}".format(hardness, p_str))
                    print("{} gold: {}".format(hardness, g_str))
                    print("")
                else:
                    turn_scores['exact'].append(1)
                scores[turn_id]['exact'] += exact_score
                scores[hardness]['exact'] += exact_score
                scores['all']['exact'] += exact_score
                for type_ in partial_types:
                    if partial_scores[type_]['pred_total'] > 0:
                        scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores[hardness]['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores[hardness]['partial'][type_]['rec_count'] += 1
                    scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                    if partial_scores[type_]['pred_total'] > 0:
                        scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores['all']['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores['all']['partial'][type_]['rec_count'] += 1
                    scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

                entries.append({
                    'predictSQL': p_str,
                    'goldSQL': g_str,
                    'hardness': hardness,
                    'exact': exact_score,
                    'partial': partial_scores
                })

        if all(v == 1 for v in turn_scores["exec"]):
            scores['joint_all']['exec'] += 1

        if all(v == 1 for v in turn_scores["exact"]):
            scores['joint_all']['exact'] += 1

    for turn in turns:
        if scores[turn]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[turn]['exec'] /= scores[turn]['count']

        if etype in ["all", "match"]:
            scores[turn]['exact'] /= scores[turn]['count']

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                                scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    print_scores(scores, etype, include_turn_acc=include_turn_acc)
    return scores


# Rebuild SQL functions for value evaluation
def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


# def rebuild_condition_val(sql):
#     for condition:
#         res.append(rebuild_cond_unit_val(condition))
#     return res

def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def remove_val(expression):
    """
    Removes values from conditions, does so in the left and right of an intersect for example. Does nothing else. Misleading name.
    """

    def transform(node):
        if isinstance(node, Literal):
            return None
        return node

    expression = expression.transform(transform)
    return expression
    # count = 0
    # for node in expression.walk(
    #         bfs=False, prune=stop_traversing
    # ):
    #     if isinstance(node, Subquery):
    #         count += 1
    #
    # return count


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql


def build_valid_col_units(sql, schema):
    """
    Return all valid col units that might be used.

    :param sql: Parsed SQL as a tree
    """
    select = sql
    while any(isinstance(select, expr) for expr in SQL_EXPR):
        select = select.left
    if not select.args.get("from"):
        return []
    from_expr = select.args["from"]
    tables = [from_expr.name]
    join_expr = select.args["joins"] if "joins" in select.args else []
    for join in join_expr:
        tables.append(join.this.name)
    prefixs = [
        f"__{table.lower()}" for table in tables
    ]
    valid_col_units = []
    for value in schema.id_map.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


# def rebuild_from_col(sql, kmap):
#     # Replaces foreign keys with corresponding column in original table, sets distinct to none if disable distinct


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in
                            from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    """
    Replaces foreign keys in the sql with the corresponding original names
    Ignores distinct (by turning it to None in Select) if disable_distinct is set
    """
    # Drop distinct nodes
    if sql is None:
        return sql

    def ignore_distinct(node):
        if isinstance(node, Distinct) and not isinstance(node.parent, Count):
            return None
        # Treat count distinct as count
        elif isinstance(node, Distinct) and isinstance(node.parent, Count):
            node.parent.set("expressions", node.expressions)
            return None
        return node

    if DISABLE_DISTINCT:
        sql = sql.transform(ignore_distinct)

    # Get columns
    root = build_scope(sql)
    for scope in root.traverse():
        alias_to_table = scope.selected_sources
        for column in scope.find_all(Column):
            # Kmap-Format
            # If table is not in alias_to_table, we can not replace the column
            if not column.table in alias_to_table:
                continue
            column_str = f"__{alias_to_table[column.table][0].name}." + column.name.lower() + "__"
            if column_str in valid_col_units and column_str in kmap:
                # Update column identifier + table accordingly
                new_column_str = kmap[column_str]
                column.this.set("this", new_column_str.split(".")[1].removesuffix("__"))
                column.args["table"].set("this", new_column_str.split(".")[0].removeprefix("__"))
            # Replace alias with original table name
            elif column_str in valid_col_units:
                column.this.set("this", column_str.split(".")[1].removesuffix("__"))
                column.args["table"].set("this", column_str.split(".")[0].removeprefix("__"))

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to id_map in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, help="the path to the gold queries")
    parser.add_argument('--pred', dest='pred', type=str, help="the path to the predicted queries")
    parser.add_argument('--db', dest='db', type=str,
                        help="the directory that contains all the databases and test suites")
    parser.add_argument('--table', dest='table', type=str, help="the tables.json schema file")
    parser.add_argument('--etype', dest='etype', type=str, default='exec',
                        help="evaluation type, exec for test suite accuracy, match for the original exact set match accuracy",
                        choices=('all', 'exec', 'match'))
    parser.add_argument('--plug_value', default=False, action='store_true',
                        help='whether to plug in the gold value into the predicted query; suitable if your model does not predict values.')
    parser.add_argument('--keep_distinct', default=False, action='store_true',
                        help='whether to keep distinct keyword during evaluation. default is false.')
    parser.add_argument('--progress_bar_for_each_datapoint', default=False, action='store_true',
                        help='whether to print progress bar of running test inputs for each datapoint')
    args = parser.parse_args()

    # only evaluting exact match needs this argument
    kmaps = None
    if args.etype in ['all', 'match']:
        assert args.table is not None, 'table argument must be non-None if exact set match is evaluated'
        kmaps = build_foreign_key_map_from_json(args.table)

    evaluate(args.gold, args.pred, args.db, args.etype, kmaps, args.plug_value, args.keep_distinct,
             args.progress_bar_for_each_datapoint)
