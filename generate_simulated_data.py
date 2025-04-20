from src.simulate_lib import simulate_multiaccum
import pandas as pd
import argparse

parser.add_argument("--truefreq", 
                    type=float, 
                    required=True, 
                    help="True (simulated) frequency"))

parser.add_argument("--readnoise",
                    type=int, 
                    required=True, 
                    help="Standard deviation of simulated read noise"
)

args = parser.parse_args()

true_freq = args.truefreq
read_noise = args.readnoise

num_measurements = 101 # Includes measurements at exactly t=0.0 and t=1.0.
num_samples = 10000 # Generally 10,000

data =[]
for i in range(num_samples):
    curr_data = simulate_multiaccum(freq=true_freq, num_measurements=num_measurements, read_noise=read_noise)
    data.append(curr_data)

df = pd.DataFrame(data).T
print(df)


df.to_csv(f"data/poisson_simulations/freq_{freq}_read_noise_{read_noise}/simulations.csv")
