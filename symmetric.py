import random
import math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def calculate_pathloss(d_TX_RX, a=61.4, b=2, sigma=5.8):
    """Calculate path loss with shadow fading in dB."""
    pathloss_dB = a + 10 * b * np.log10(d_TX_RX) + sigma * np.random.randn()
    pathloss_linear = 10 ** ((pathloss_dB - 30) / 10)  # Convert to Watts
    return pathloss_linear

def generate_rician_channel(Nr, Nt, pathloss_linear, epsilon=2, L=2):
    """Generate Rician fading channel matrix with LoS and NLoS components."""
    # Line-of-Sight (LoS) component
    H_LoS = np.sqrt(1 / 2) * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))

    # Non-Line-of-Sight (NLoS) component
    H_NLoS = np.zeros((Nr, Nt), dtype=complex)
    for _ in range(L):
        H_NLoS += np.sqrt(1 / 4) * (
            np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)
        )

    # Combine LoS and NLoS components
    H = np.sqrt(1 / pathloss_linear) * (
        np.sqrt(epsilon / (1 + epsilon)) * H_LoS + np.sqrt(1 / (1 + epsilon)) * H_NLoS
    )
    return H

def rand_coor_t(center_x, center_y, radius):
    theta = random.uniform(0, math.pi) 
    
    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)

    return round(x, 2), round(y, 2)

def rand_coor_r(center_x, center_y, radius):
    theta = random.uniform(math.pi, 2*math.pi) 
    
    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)

    return round(x, 2), round(y, 2)

def channel_gain_r_user(chromosome, N):
    r_x, r_y=rand_coor_r(0,50,3)
    TX_xy=np.array([0,0])
    RX_xy=np.array([r_x,r_y])
    d_TX_RX = np.linalg.norm(TX_xy - RX_xy)

    d_k=generate_rician_channel(1,1,calculate_pathloss(d_TX_RX))
    v_k=generate_rician_channel(math.isqrt(N),1,calculate_pathloss(3))
    g=generate_rician_channel(math.isqrt(N),1,calculate_pathloss(50))
    
    reflection_matrix = np.zeros((math.isqrt(N), math.isqrt(N)), dtype=complex)
    
    for i in range(N):
        beta_t = chromosome[2 * i]
        _, theta_r = chromosome[2 * i + 1]  
        
        beta_r = 1 - beta_t  
        reflection_coefficient = np.sqrt(beta_r) * np.exp(1j * theta_r) 

        row, col = divmod(i, math.isqrt(N)) 
        reflection_matrix[row, col] = reflection_coefficient

    c_k = d_k + np.matmul(np.matmul(v_k.conj().T, reflection_matrix), g)
    
    return abs(c_k)

def channel_gain_t_user(chromosome, N):
    t_x, t_y=rand_coor_t(0,50,3)
    TX_xy=np.array([0,0])
    RX_xy=np.array([t_x,t_y])
    d_TX_RX = np.linalg.norm(TX_xy - RX_xy)

    d_k=generate_rician_channel(1,1,calculate_pathloss(d_TX_RX))
    v_k=generate_rician_channel(math.isqrt(N),1,calculate_pathloss(3))
    g=generate_rician_channel(math.isqrt(N),1,calculate_pathloss(50))

    transmission_matrix = np.zeros((math.isqrt(N), math.isqrt(N)), dtype=complex)


    for i in range(N):
        beta_t=chromosome[2*i]
        theta_t, _=chromosome[2*i+1]

        transmission_coefficient=np.sqrt(beta_t)*np.exp(1j*theta_t)

        row, col=divmod(i,math.isqrt(N))
        transmission_matrix[row,col]=transmission_coefficient

    c_k = d_k + np.matmul(np.matmul(v_k.conj().T, transmission_matrix), g)
    
    return abs(c_k)

def rate_oma(chromosome, N, user):
    channel_gain_t=channel_gain_t_user(chromosome, N)
    channel_gain_r=channel_gain_r_user(chromosome, N)
    power_t=chromosome[2*N]
    power_r=chromosome[2*N+1]
    x=1 + 2 * abs(channel_gain_t)*abs(channel_gain_t) * power_t / var_n
    y=1 + 2 * abs(channel_gain_r)*abs(channel_gain_r) * power_r / var_n
    return (
        (0.5)*np.log2(x)
        if user == "t"
        else (0.5)*np.log2(y)
    )

def rate_noma(chromosome, N, user):
    channel_gain_t=channel_gain_t_user(chromosome, N)
    channel_gain_r=channel_gain_r_user(chromosome, N)
    power_t=chromosome[2*N]
    power_r=chromosome[2*N+1]
    x=1 + ((abs(channel_gain_t) ** 2) * power_t) / (var_n)
    y=1 + ((abs(channel_gain_r) ** 2) * power_r) / ((abs(channel_gain_r) ** 2) * power_t + var_n)
    return (
        np.log2(x)
        if user == "t"
        else np.log2(y)
    )


def fitness_function_oma(chromosome, N):
    p_t, p_r = chromosome[2*N], chromosome[2*N+1]
    # print(f"The value of p_t is {p_t} and p_r is {p_r}")
    
    rate_t = rate_oma(chromosome, N, "t")
    rate_r = rate_oma(chromosome, N, "r")
    # print(f"rate t: {rate_t}, rate_r: {rate_r}")
    fitness = -(p_t + p_r) 

    if rate_t<rate_req_t:
        fitness -= penalty 
    if rate_r < rate_req_r:
        fitness -= penalty 
    
    return fitness

def fitness_function_noma(chromosome, N):
    p_t, p_r = chromosome[2*N], chromosome[2*N+1]
    # print(f"The value of p_t is {p_t} and p_r is {p_r}")
    
    rate_t = rate_noma(chromosome, N, "t")
    rate_r = rate_noma(chromosome, N, "r")
    # print(f"rate t: {rate_t}, rate_r: {rate_r}")
    fitness = -(p_t + p_r) 

    if rate_t<rate_req_t:
        fitness -= penalty 
    if rate_r < rate_req_r:
        fitness -= penalty 
    
    return fitness

def phase_constraint(theta):
    if theta < 2 * math.pi:
        phi_t = theta
        phi_r = theta + math.pi / 2
        if phi_r > 2 * math.pi:
            phi_r -= 2 * math.pi
    else:
        phi_t = theta - 2 * math.pi
        phi_r = theta - math.pi / 2
        if phi_r > 2 * math.pi:
            phi_r -= 2 * math.pi
    return (phi_t, phi_r)

def create_initial_population(size, N):
    population = []
    for _ in range(size):
        chromosome = []

        for _ in range(N):
            beta_t = random.uniform(0, 1)
            chromosome.append(beta_t)
            theta = random.uniform(0, 4 * math.pi)
            chromosome.append(phase_constraint(theta))

        p_t = random.uniform(0.5, 10)
        p_r = random.uniform(0.5, 10)
        chromosome.append(p_t)
        chromosome.append(p_r)

        population.append(chromosome)

    return population

def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Use uniform crossover
def crossover(parent1, parent2, N):
    crossover_mask = [random.choice([0, 1]) for _ in range(N)]

    offspring1 = []
    offspring2 = []

    for i in range(N):
        if crossover_mask[i] == 1: # Swap
            offspring1.append(parent2[2*i])  
            offspring1.append(parent2[2*i+1])
            offspring2.append(parent1[2*i])  
            offspring2.append(parent1[2*i+1])  
        else: # Keep 
            offspring1.append(parent1[2*i]) 
            offspring1.append(parent1[2*i+1])  
            offspring2.append(parent2[2*i]) 
            offspring2.append(parent2[2*i+1])  

    # Add p_t and p_r 
    offspring1.append(parent2[2*N])  
    offspring1.append(parent2[2*N+1])  
    offspring2.append(parent1[2*N])  
    offspring2.append(parent1[2*N+1])  

    return offspring1, offspring2

def mutation(chromosome, N, mutation_rate=0.1):
    # Loop over each element in the chromosome
    for i in range(2*N):
        if random.random() < mutation_rate:
            if isinstance(chromosome[i], tuple):  # If element is a pair.
                l=list(chromosome[i])
                l[0]=random.uniform(0, 2*math.pi)
                l[1]=chromosome[i][0]+math.pi/2
                if l[1]>2*math.pi:
                    l[1]-=2*math.pi
                chromosome[i]=tuple(l)
            else:
                chromosome[i] = random.uniform(0, 1)  
    if(random.random()<mutation_rate):
        if random.random()<0.5:
            chromosome[2*N]*=random.uniform(0.6, 1.3)
        else:
            chromosome[2*N+1]*=random.uniform(0.6, 1.3)
    return chromosome

def terminated_condition_oma(population, N):
    penalized_cnt=0
    fitnesses = [fitness_function_oma(ind, N) for ind in population]
    for i in fitnesses:
        if i<-penalty:
            penalized_cnt+=1
    return penalized_cnt>=(len(fitnesses)*0.2)

def terminated_condition_noma(population, N):
    penalized_cnt=0
    fitnesses = [fitness_function_noma(ind, N) for ind in population]
    for i in fitnesses:
        if i<-penalty:
            penalized_cnt+=1
    return penalized_cnt>=(len(fitnesses)*0.3)

def genetic_algorithm_oma(population_size, generations, N):
    population = create_initial_population(population_size, N)

    table = PrettyTable()
    table.field_names = ["Generation", "Fitness_OMA"]

    fitness_history = []  

    best_individual=[]
    for generation in range(generations):
        if terminated_condition_oma(population, N) is True:
            break
        fitnesses=[]
        for i in population:
            fitnesses.append(fitness_function_oma(i,N))

        best_idx=fitnesses.index(max(fitnesses))
        best_individual=population[best_idx]
        best_fitness = fitnesses[best_idx]

        fitness_history.append(best_fitness)
        table.add_row([generation + 1, best_fitness])

        population = selection(population, fitnesses)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            crossover_number = random.random()
            if crossover_number < 0.9:
                child1, child2 = crossover(parent1, parent2, N)
            else:
                child1, child2 = parent1, parent2
            next_population.append(mutation(child1, N))
            next_population.append(mutation(child2, N))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population
    print(table)
    if len(best_individual)!=0:
        total_power_consumption=best_individual[2*N]+best_individual[2*N+1]
    else:
        total_power_consumption=0.5
    return total_power_consumption, fitness_history

def genetic_algorithm_noma(population_size, generations, N):
    population = create_initial_population(population_size, N)

    # Prepare for table
    table = PrettyTable()
    table.field_names = ["Generation", "Fitness_NOMA"]

    fitness_history = []  # To store best fitness of each generation

    best_individual=[]
    for generation in range(generations):
        if terminated_condition_noma(population, N) is True:
            # print(f"Terminated at gereration: {generation+1}")
            break
        fitnesses=[]
        for i in population:
            fitnesses.append(fitness_function_noma(i,N))

        best_idx=fitnesses.index(max(fitnesses))
        best_individual=population[best_idx]

        best_fitness = fitnesses[best_idx]

        # rate_t = rate_noma(best_individual, N, "t")
        # rate_r = rate_noma(best_individual, N, "r")
        # print(f"rate t: {rate_t}, rate_r: {rate_r}")

        fitness_history.append(best_fitness)
        table.add_row([generation + 1, best_fitness])

        population = selection(population, fitnesses)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            crossover_number = random.random()
            if crossover_number < 0.9:
                child1, child2 = crossover(parent1, parent2, N)
            else:
                child1, child2 = parent1, parent2
            next_population.append(mutation(child1, N))
            next_population.append(mutation(child2, N))

        next_population[0] = best_individual
        population = next_population
    print(table)
    # print(f"size of best individual: {len(best_individual)}")
    if len(best_individual)!=0:
        total_power_consumption=best_individual[2*N]+best_individual[2*N+1]
    else:
        total_power_consumption=0.5
    return total_power_consumption, fitness_history

if __name__ == "__main__":
    var_n = 10 ** ((-80 - 30) / 10)  # Noise power in Watts

    # N = 16  # 16 elemets in one RIS
    penalty = 100

    population_size = 100
    generations = 80

    N_values=[n**2 for n in range(3, 8)]
    number_of_realization=6

    power_for_plot = []
    fitness_histories={N: [] for N in N_values}

    # Rate requirements in symmetric case
    # rate_req_r = 5
    # rate_req_t = 5
    # rate_requirements=[0.4,0.5,0.6,0.7,0.8]
    rate_requirements=[0.5,0.75,1,1.5,2,2.5,3]
    power_vs_rate_oma=[]
    power_vs_rate_noma=[]

    for rates in rate_requirements:
        rate_req_r=rates
        rate_req_t=rates

        # oma case
        power_for_rate_oma=0
        for _ in range(number_of_realization):
            p_total, _=genetic_algorithm_oma(population_size,generations,16)
            power_for_rate_oma+=p_total

        power_avg=power_for_rate_oma/number_of_realization
        power_vs_rate_oma.append(power_avg)

        # noma case
        power_for_rate_noma=0
        for _ in range(number_of_realization):
            p_total, _=genetic_algorithm_noma(population_size,generations,16)
            power_for_rate_noma+=p_total

        power_avg=power_for_rate_noma/number_of_realization
        power_vs_rate_noma.append(power_avg)


    _, fitness_history_oma = genetic_algorithm_oma(population_size, generations, 16)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history_oma) + 1), fitness_history_oma, marker="o", linestyle="-", color="g", label="OMA")
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Best Fitness Value", fontsize=14)
    plt.title("Fitness Value vs Generation for OMA", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(rate_requirements, power_vs_rate_oma, marker="o", linestyle="-", color="g", label="Average Power Consumption")
    plt.xlabel("Rate Requirements (bit/s/Hz)", fontsize=14)
    plt.ylabel("Average Power Consumption (Watts)", fontsize=14)
    plt.title("Power Consumption vs Rate Requirements(OMA)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

   
    # Plotting Power vs Rate Requirements
    plt.figure(figsize=(8, 6)) 
    # OMA
    plt.plot(rate_requirements, power_vs_rate_oma, 
            marker="o", linestyle="-", color="g", label="Avg Power Consumption for OMA")
    # NOMA
    plt.plot(rate_requirements, power_vs_rate_noma, 
            marker="o", linestyle="-", color="b", label="Avg Power Consumption for NOMA")

    plt.xlabel("Rate Requirements (bit/s/Hz)", fontsize=14)
    plt.ylabel("Average Power Consumption (Watts)", fontsize=14)
    plt.title("Power Consumption vs Rate Requirements", fontsize=16)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend() 
    plt.tight_layout()
    plt.show()
    

    # Plotting Fitness vs Generation for each N
    # plt.figure(figsize=(10, 6))
    # for N, avg_fitness_history in fitness_histories.items():
    #     plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history, label=f"N={N}")
    # plt.xlabel("Generation", fontsize=14)
    # plt.ylabel("Average Fitness Value", fontsize=14)
    # plt.title("Fitness Value vs Generation for Different N", fontsize=16)
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()