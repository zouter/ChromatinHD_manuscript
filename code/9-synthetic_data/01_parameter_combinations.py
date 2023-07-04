#%%
import random
import pickle

# n_values, shift_y_density, noise, shift_slope, p, fragment_length

gene_parameters_list_manual = [
    [
        [1000, 0, 20, 0, 0, 150], 
        [1000, -2, 20, 0, 5000, 150],
        [1000, 2, 20, 0, -5000, 150] 
    ],
    [
        [1000, 0, 15, 0, 3000, 150], 
        [1000, 0, 15, 0, -3000, 150], 
    ],
]

gene_parameters_list_combi = []

# Generate 200 gene parameter lists
for _ in range(200):
    gene_parameters = []
    num_lists = random.randint(2, 5)  # Random number of lists within the range 2-10

    # Generate 'num_lists' lists
    for _ in range(num_lists):
        n_values = random.randint(200, 2000)
        shift_y_density = random.uniform(-2, 2)
        noise = random.uniform(0, 2)
        shift_slope = random.uniform(0, 0.1)
        p = random.randint(-10000, 10000)
        fragment_length = random.randint(125, 150)

        gene_parameters.append([n_values, shift_y_density, noise, shift_slope, p, fragment_length])

    gene_parameters_list_combi.append(gene_parameters)

gene_parameters_list = gene_parameters_list_manual + gene_parameters_list_combi

#%%
# Save gene_parameters_list as a pickle file
pickle.dump(gene_parameters_list, open('gene_parameters_list.pickle', 'wb'))

#%%
