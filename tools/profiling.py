import pstats

p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(10)  # Print top 10 functions by cumulative time