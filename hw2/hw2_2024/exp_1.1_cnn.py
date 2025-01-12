import os
from hw2.experiments import cnn_experiment

if __name__ == '__main__':

    seed = 42

    K = [32 , 64]
    L= [2,4,8,16]

    print("Starting Experiments ")
    if not os.path.exists('results'):
        os.mkdir('results')
    for (l,k) in zip(L,K):
        # for every choice invoke cnn_experiment and save the results to JSON
        print(f" K = {k} , L= {l}") 
        run_name = f"exp1_1_L{l}_K{k}.json"
        cnn_experiment(
            run_name, seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=[k], layers_per_block=l, pool_every=10, hidden_dims=[100],
            model_type='resnet',
            pooling_params= dict(kernel_size=2)
        )
    print("done")


