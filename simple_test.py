import multiview as mv

MvSR = mv.MvSR("datasets/linear/", max_length=10, pop_size=100, generations=100, n_params=3)
MvSR.run()
result_table = MvSR.raw_results
MvSR.plot_all_fits()
