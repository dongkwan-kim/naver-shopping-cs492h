import argparse
import numpy as np
import os
import pickle
import time

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from data_loader import get_loader
from models import CategoryClassifier

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if IS_ON_NSML:
        args.model_path = os.path.join(DATASET_PATH, 'train', args.model_path)
        args.csv_path = os.path.join(DATASET_PATH, 'train', args.csv_path)
        args.vocab_path = os.path.join(DATASET_PATH, 'train', args.vocab_path)
        args.image_dir = os.path.join(DATASET_PATH, 'train', args.image_dir)

    # Image preprocessing, normalization for the pretrained network
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_loader, test_loader, label_enc, data_size= get_loader(args.csv_path, args.image_dir, args.batch_size,
                                                      transform, vocab, args.max_len, args.num_workers,
                                                      shuffle=True, test_split=0.1)

    # Build the models
    vocab_size = len(vocab)
    num_class = len(label_enc.classes_)

    model = CategoryClassifier(args, vocab_size, num_class)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    start = time.time()
    best_acc = 0.0

    for epoch in range(args.num_epochs):

        # Each epoch has a training and test phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
                dataset_sizes = data_size['train']
                total_step = len(train_loader)
            else:
                model.eval()  # Set model to evaluate mode
                data_loader = test_loader
                dataset_sizes = data_size['test']
                total_step = len(test_loader)

            running_loss = 0.0
            running_corrects = 0

            for i, batch in enumerate(data_loader):
                # Set mini-batch dataset
                images = batch['image'].to(device)
                texts = batch['text'].to(device)
                labels = batch['label'].view(-1).to(device)

                # Resets the gradient to 0
                optimizer.zero_grad()

                # Forward, backward and optimize
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, texts)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Print log info
                width = 20
                recv_per = int(100 * (i + 1) / total_step)
                show_bar = ('[%%-%ds]' % width) % (int(width * recv_per / 100) * ">")
                use_time = time.time() - start
                acc = (preds == labels).sum().item() / labels.size(0)

                show_str = '[Epoch %d/%d] %d/%d %s -%.1fs/step - loss: %.4f - acc: %.4f '
                print(
                    show_str % (epoch, args.num_epochs, i + 1, total_step, show_bar, use_time, loss.item(), acc),
                    end='\n'
                )
            if phase == 'train':
                scheduler.step()
            if phase == 'val':
                epoch_loss = running_loss / dataset_sizes
                epoch_acc = running_corrects.double() / dataset_sizes
                print('{} Loss({}): {:.4f} Acc: {:.4f}'.format(phase, dataset_sizes, epoch_loss, epoch_acc))


            # TODO: Add validation accuracy, loss
            nsml.report(
                summary=True,
                step=epoch,
                **{'train__acc': acc, 'train__loss': loss.item()},
            )

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            # Set mini-batch dataset
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].view(-1).to(device)

            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--csv_path', type=str, default='./data/fashion_dataset.csv', help='path for dataset file')
    parser.add_argument('--crop_size', type=int, default=224, help='size for cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/image', help='directory for resized images')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Data parameters
    parser.add_argument('--max_len', type=int, default=20, help='maximum length of a text')

    # Model parameters
    parser.add_argument('--encoded_image_size', type=int, default=3, help='resize image to fixed size')
    parser.add_argument('--embed_dim', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--window_sizes', type=int, default=[3, 4, 5], help='window_sizes')
    parser.add_argument('--n_filters', type=int, default=100, help='number of filter')
    parser.add_argument('--dropout_p', type=int, default=0.5, help='dropout ratio')
    parser.add_argument('--static', type=int, default=False, help='static embedding or not')

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    args = parser.parse_args()
    print(args)
    main(args)
