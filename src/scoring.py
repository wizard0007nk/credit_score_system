def generate_credit_score(prob: float) -> int:
    # Scale probability to a score between 300 and 850
    return int(300 + prob * 550)
