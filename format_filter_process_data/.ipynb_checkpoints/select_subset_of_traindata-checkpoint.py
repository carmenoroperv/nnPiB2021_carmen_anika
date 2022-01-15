import torch

train_seq_full = torch.load(snakemake.input.full_train_seq)
train_target_full = torch.load(snakemake.input.full_train_target)

print("Full traindata size: ")
print(train_seq_full.shape[0])


indices_20 = [*range(0, train_seq_full.shape[0], 5)]
indices_50 = [*range(0, train_seq_full.shape[0], 2)]
indices_80 = []

for i in range(0, train_seq_full.shape[0]):
    if i % 5 == 0 and i != 0:
        continue
    indices_80.append(i)
    
print("Indices to choose: 20%")
print(indices_20[0:6])
print("Indices to choose: 50%")
print(indices_50[0:6])
print("Indices to choose: 80%")
print(indices_80[0:6])

train_seq_20  = train_seq_full[indices_20]
train_seq_50  = train_seq_full[indices_50]
train_seq_80  = train_seq_full[indices_80]

train_target_20  = train_target_full[indices_20]
train_target_50  = train_target_full[indices_50]
train_target_80  = train_target_full[indices_80]

del train_seq_full
del train_target_full

print("Subset of 20% - number of rows and percentage of full data")
print(train_seq_20.shape[0])
print(train_target_20.shape[0])
print(train_seq_20.shape[0]*100/11410422)

print("Subset of 50% - number of rows and percentage of full data")
print(train_seq_50.shape[0])
print(train_target_50.shape[0])
print(train_seq_50.shape[0]*100/11410422)

print("Subset of 80% - number of rows and percentage of full data")
print(train_seq_80.shape[0])
print(train_target_80.shape[0])
print(train_seq_80.shape[0]*100/11410422)

torch.save(train_seq_20, snakemake.output.output_seq_20)
torch.save(train_target_20, snakemake.output.output_target_20)

del train_seq_20
del train_target_20

torch.save(train_seq_50, snakemake.output.output_seq_50)
torch.save(train_target_50, snakemake.output.output_target_50)

del train_seq_50
del train_target_50

torch.save(train_seq_80, snakemake.output.output_seq_80)
torch.save(train_target_80, snakemake.output.output_target_80)

#run train_model_fully_connected_one_layer.py with snakefile_tune_linear_one_layer_LR_MANUAL to evaluate which subset is best