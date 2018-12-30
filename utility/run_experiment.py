def run_experiment(df, benchmark):
    sentences1 = df['sent_1']
    sentences2 = df['sent_2']

    sims = benchmark[1](sentences1, sentences2)

    return sims, benchmark[0]
