import torch
import time
import copy


def train_model(model, dataloaders, criterion, optimizer,
                epochs, device, scheduler=None, half=False, to_long=False):
    since = time.time()
    losses = {phase: [] for phase in dataloaders}
    accs = {phase: [] for phase in dataloaders}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")
        print("="*10)

        for phase in dataloaders:
            loader = dataloaders[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_acc = 0.0, 0.0
            with torch.set_grad_enabled(phase == "train"):
                for inputs, labels in loader:
                    if half:
                        inputs = inputs.half()
                    inputs, labels = inputs.to(device), labels.to(device)
                    if to_long:
                        inputs = inputs.long()
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    _, preds = torch.topk(outputs, 1, dim=1)
                    loss = criterion(outputs, labels.flatten())
                    running_loss += loss.item() * len(inputs)
                    running_acc += torch.sum(preds == labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            ds_size = len(loader.dataset)
            epoch_loss, epoch_acc = running_loss/ds_size, running_acc.double()/ds_size
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.5f}\tAcc:{epoch_acc:.5f}\n")

            if phase == "valid":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()

    time_elapsed =  time.time() - since
    print(f"training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, losses, accs


def train_model_prottrans(model, dataloaders, criterion, optimizer, epochs,
                          device, embedder, scheduler=None, half=False, to_long=False):
    since = time.time()
    losses = {phase: [] for phase in dataloaders}
    accs = {phase: [] for phase in dataloaders}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")
        print("="*10)

        for phase in dataloaders:
            loader = dataloaders[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_acc = 0.0, 0.0
            with torch.set_grad_enabled(phase == "train"):
                for inputs, masks, labels in loader:
                    if half:
                        inputs = inputs.half()
                        masks = masks.half()
                    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                    if to_long:
                        inputs = inputs.long()
                        masks = masks.long()
                    with torch.no_grad():
                        embedding = embedder(input_ids=inputs, attention_mask=masks)[0].half()
                    optimizer.zero_grad()

                    outputs = model(embedding)
                    _, preds = torch.topk(outputs, 1, dim=1)
                    loss = criterion(outputs, labels.flatten())
                    running_loss += loss.item() * len(inputs)
                    running_acc += torch.sum(preds == labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            ds_size = len(loader.dataset)
            epoch_loss, epoch_acc = running_loss/ds_size, running_acc.double()/ds_size
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.5f}\tAcc:{epoch_acc:.5f}\n")

            if phase == "valid":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()

    time_elapsed =  time.time() - since
    print(f"training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, losses, accs
