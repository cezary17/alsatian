import torch

CUDA = 'cuda'


# code adapted from/inspired by: https://github.com/DS3Lab/shift/blob/1db7f15d5fe4261d421f96c1b3a92492c8ca6b07/server/worker_general/general/classifier/_linear.py

def get_input_dimension(batch):
    sample_tensor: torch.Tensor = batch[0]
    return sample_tensor.shape


def linear_proxy(train_features, train_labels, test_features, test_labels, num_classes, device):
    first_item = train_features[0]
    input_dimension = first_item.shape

    # init objects
    model = torch.nn.Linear(input_dimension[1], num_classes)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss_function = torch.nn.CrossEntropyLoss().to(device)

    # train model on train data
    model.train()
    for i in range(100):
        for feature_batch, label_batch in zip(train_features, train_labels):
            optimizer.zero_grad()
            outputs = model(feature_batch)
            label_batch = label_batch.to(device)
            loss = loss_function(outputs, label_batch)
            loss.backward()
            optimizer.step()

    print('done training')

    # eval model on test data
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    # loss_func = torch.nn.CrossEntropyLoss()

    loss_function.to(device)

    for feature_batch, label_batch in zip(test_features, test_labels):
        outputs = model(feature_batch)
        # loss = loss_func(outputs, label_batch)
        label_batch = label_batch.to(device)

        loss = loss_function(outputs, label_batch)

        total_loss += loss.item()

        # Calculate top-1 accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += label_batch.size(0)
        correct_predictions += (predicted == label_batch).sum().item()

    average_loss = total_loss / len(test_features)
    top1_accuracy = correct_predictions / total_samples

    print('done inference')
    return average_loss, top1_accuracy


def collect_features_and_labels(caching_service, device, train_feature_ids, train_label_ids, bert_features):
    features = []
    labels = []
    for feature_id, label_id in zip(train_feature_ids, train_label_ids):
        feature_batch = caching_service.get_item(feature_id)
        label_batch = caching_service.get_item(label_id)
        if bert_features:
            last_hidden_state = feature_batch[0]
            cls_hidden_state = last_hidden_state[:, 0, :]
            feature_batch = cls_hidden_state
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        feature_batch, label_batch = torch.squeeze(feature_batch), torch.squeeze(label_batch)
        features.append(feature_batch)
        labels.append(label_batch)
    return features, labels
