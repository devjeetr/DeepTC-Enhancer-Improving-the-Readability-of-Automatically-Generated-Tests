from preprocessing.context_utils import (
    mask_variables_in_context,
    mask_variables_in_contexts,
)


def test_mask_variables_in_context_doesnt_mask_non_variables():
    context = ("a", "VDID|Cal0|Nm0", "b")

    start, path, end = mask_variables_in_context(context, ["a"])

    assert start == "VARIABLE"
    assert end == context[-1]
    assert path == context[1]


def test_mask_variables_in_context_masks_end():
    context = ("a", "VDID|Cal0|Nm0", "b")

    start, path, end = mask_variables_in_context(context, ["a", "b"])

    assert start == "VARIABLE"
    assert path == context[1]
    assert end == "VARIABLE"


def test_mask_variables_in_context_with_target_variable():
    context = ("a", "VDID|Cal0|Nm0", "b")

    start, path, end = mask_variables_in_context(
        context, ["a", "b"], target_variable="a"
    )

    assert start == "TARGET_VARIABLE"
    assert path == context[1]
    assert end == "VARIABLE"


def test_mask_variables_in_contexts_with_target_variable():

    context_a = ("a", "VDID|Cal0|Nm0", "b")
    context_b = ("b", "VDID|Cal0|Nm0", "a")

    context_a, context_b = mask_variables_in_contexts(
        [context_a, context_b], target_variable="a"
    )

    assert context_a[0] == "TARGET_VARIABLE"
    assert context_b[2] == "TARGET_VARIABLE"
    assert context_a[2] == "VARIABLE"
    assert context_b[0] == "VARIABLE"


def test_mask_variables_in_contexts_with_target_variable_no_variables_provided():

    context_a = ("a", "VDID|Cal0|Nm0", "b")
    context_b = ("b", "VDID|Cal0|Nm0", "a")

    context_a, context_b = mask_variables_in_contexts(
        [context_a, context_b], target_variable="a", variables=["a", "b"]
    )

    assert context_a[0] == "TARGET_VARIABLE"
    assert context_b[2] == "TARGET_VARIABLE"
    assert context_a[2] == "VARIABLE"
    assert context_b[0] == "VARIABLE"
