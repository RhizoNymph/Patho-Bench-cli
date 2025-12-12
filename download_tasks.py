import datasets
dataset='*'
datasets.load_dataset(
                    'MahmoodLab/Patho-Bench', 
                    cache_dir='./tasks',
                    dataset_to_download=dataset,
                    trust_remote_code=True
                )
