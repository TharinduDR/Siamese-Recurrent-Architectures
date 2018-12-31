def run_experiment(train_df, test_df, sent_cols, sim_col, benchmark):
    sims, trained_model = benchmark[1](train_df, test_df, sent_cols, sim_col)
    return sims, trained_model, benchmark[0]
