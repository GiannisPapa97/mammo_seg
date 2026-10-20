import pytest

from mammo_seg.data.splitting import Patient, hold_out, stratified_folds


def make_patients(with_mass: int, without_mass: int) -> list[Patient]:
    return [
        Patient(patient_id=f"p{index:03d}", has_mass=index < with_mass)
        for index in range(with_mass + without_mass)
    ]


def positive_ids(patients: list[Patient]) -> set[str]:
    return {patient.patient_id for patient in patients if patient.has_mass}


# --- folds ---


def test_folds_partition_the_patients() -> None:
    patients = make_patients(20, 30)
    folds = stratified_folds(patients, 5, seed=0)
    assert len(folds) == 5
    assert all(folds)
    flat = [patient_id for fold in folds for patient_id in fold]
    assert len(flat) == len(set(flat)) == 50
    assert set(flat) == {patient.patient_id for patient in patients}


def test_folds_keep_the_share_of_patients_with_a_mass() -> None:
    patients = make_patients(50, 58)  # the real dataset: 50 positive, 58 negative
    folds = stratified_folds(patients, 5, seed=0)
    positive = positive_ids(patients)
    assert sum(len(fold) for fold in folds) == 108
    assert {len(fold) for fold in folds} <= {21, 22}
    assert all(len(positive & set(fold)) == 10 for fold in folds)
    assert all(11 <= len(set(fold) - positive) <= 12 for fold in folds)


def test_folds_and_the_ids_inside_are_sorted() -> None:
    for fold in stratified_folds(make_patients(20, 30), 4, seed=3):
        assert list(fold) == sorted(fold)


def test_the_same_seed_gives_the_same_folds() -> None:
    patients = make_patients(20, 30)
    assert stratified_folds(patients, 5, seed=7) == stratified_folds(
        patients, 5, seed=7
    )


def test_different_seeds_give_different_folds() -> None:
    patients = make_patients(20, 30)
    assert stratified_folds(patients, 5, seed=0) != stratified_folds(
        patients, 5, seed=1
    )


def test_the_order_of_the_patients_does_not_matter_for_folds() -> None:
    patients = make_patients(20, 30)
    assert stratified_folds(patients, 5, seed=1) == stratified_folds(
        patients[::-1], 5, seed=1
    )


def test_a_set_with_a_single_class_can_still_be_split_into_folds() -> None:
    folds = stratified_folds(make_patients(0, 12), 3, seed=0)
    assert sorted(len(fold) for fold in folds) == [4, 4, 4]


@pytest.mark.parametrize("n_folds", [1, 0, -3])
def test_fewer_than_two_folds_are_rejected(n_folds: int) -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        stratified_folds(make_patients(5, 5), n_folds, seed=0)


def test_a_class_too_small_for_the_folds_is_rejected() -> None:
    with pytest.raises(ValueError, match="one class has only 3 patient"):
        stratified_folds(make_patients(3, 30), 5, seed=0)


def test_more_folds_than_patients_are_rejected() -> None:
    with pytest.raises(ValueError):
        stratified_folds(make_patients(1, 2), 5, seed=0)


# --- hold-out ---


def test_hold_out_partitions_the_patients() -> None:
    patients = make_patients(20, 30)
    train, test = hold_out(patients, 0.2, seed=0)
    assert train and test
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == {patient.patient_id for patient in patients}
    assert list(train) == sorted(train)
    assert list(test) == sorted(test)


def test_hold_out_keeps_the_share_of_patients_with_a_mass() -> None:
    patients = make_patients(50, 58)
    train, test = hold_out(patients, 0.2, seed=0)
    positive = positive_ids(patients)
    assert 21 <= len(test) <= 22
    assert abs(len(positive & set(test)) - 10) <= 1
    assert abs(len(positive & set(train)) - 40) <= 1


def test_hold_out_is_reproducible_and_ignores_the_input_order() -> None:
    patients = make_patients(20, 30)
    expected = hold_out(patients, 0.2, seed=7)
    assert hold_out(patients, 0.2, seed=7) == expected
    assert hold_out(patients[::-1], 0.2, seed=7) == expected
    assert hold_out(patients, 0.2, seed=8) != expected


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_a_fraction_outside_zero_and_one_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        hold_out(make_patients(5, 5), fraction, seed=0)


def test_a_class_too_small_for_the_hold_out_is_rejected() -> None:
    with pytest.raises(ValueError, match="stratified on masses"):
        hold_out(make_patients(1, 30), 0.2, seed=0)
    with pytest.raises(ValueError, match="stratified on masses"):
        hold_out(make_patients(50, 58), 0.001, seed=0)


# --- both ---


def test_no_patients_are_rejected() -> None:
    with pytest.raises(ValueError, match="no patients"):
        stratified_folds([], 2, seed=0)
    with pytest.raises(ValueError, match="no patients"):
        hold_out([], 0.2, seed=0)


def test_duplicate_patient_ids_are_rejected() -> None:
    patients = [*make_patients(5, 5), Patient("p000", has_mass=True)]
    with pytest.raises(ValueError, match="duplicate patient ids: \\['p000'\\]"):
        stratified_folds(patients, 2, seed=0)
    with pytest.raises(ValueError, match="duplicate patient ids"):
        hold_out(patients, 0.2, seed=0)
