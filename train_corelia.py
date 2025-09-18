from ludwig.api import LudwigModel

# Load Corelia's manifest
model = LudwigModel(config='corelia_manifest.yaml')

# Train using a placeholder dataset (Titanic)
train_stats = model.train(
    dataset='https://raw.githubusercontent.com/ludwig-ai/ludwig/main/examples/datasets/titanic.csv'
)

# Print results
print(train_stats)
