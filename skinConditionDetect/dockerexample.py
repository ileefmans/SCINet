from datahelper import CreateDataset
from torch.utils.data import DataLoader

pickle_path = "/Users/ianleefmans/Desktop/Insight/Project/Re-Identifying_Persistent_Skin_Conditions/skinConditionDetect/annotation_dict.pkl"
data_directory = "/Users/ianleefmans/Desktop/Insight/Project/Data"

# Create dataset
dataset = CreateDataset(pickle_path, data_directory, transform = torchvision.transforms.ToTensor())
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    #target = torch.LongTensor(target)
    return data, target
train_loader = DataLoader(dataset=dataset, batch_size=4, num_workers=4, shuffle=True, collate_fn=my_collate)
sample = iter(train_loader).next()


print(sample[0].size(), type(sample[1]))