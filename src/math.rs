pub fn round_up(num_to_round: usize, multiple: usize) -> usize {
    assert_ne!(multiple, 0);
    if num_to_round == 0 {
        0
    } else {
        ((num_to_round - 1) / multiple + 1) * multiple
    }
}
