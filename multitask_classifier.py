import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO

        # baseline ---
        # similarity
        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.similarity_classifier = nn.Linear(2 * config.hidden_size, 1)

        # paraphrase
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_classifier = nn.Linear(2 * config.hidden_size, 1)

        # sentiment
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(config.hidden_size, len(config.num_labels))

        ## todo extension

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        return self.bert(input_ids, attention_mask)["pooler_output"]


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooled_output = self.forward(input_ids, attention_mask)
        pooled_output = self.sentiment_dropout(pooled_output)
        logits = self.sentiment_classifier(pooled_output)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        pooled_output = torch.cat((pooled_output_1, pooled_output_2), dim=-1)
        pooled_output = self.paraphrase_dropout(pooled_output)
        logits = self.paraphrase_classifier(pooled_output)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooled_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_output_2 = self.forward(input_ids_2, attention_mask_2)
        pooled_output = torch.cat((pooled_output_1, pooled_output_2), dim=-1)
        pooled_output = self.similarity_dropout(pooled_output)
        logits = self.similarity_classifier(pooled_output)
        return logits

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def custom_iterator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    ## sst
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    ## sts
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)       
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)         

    ## para
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    iter_sts_data, iter_sst_data = (iter(custom_iterator(sts_train_dataloader)), iter(custom_iterator(sst_train_dataloader)))

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_score = 0

    def extract_labels(batch, single_label):
        if single_label:
            return (batch['token_ids'].to(device), batch['attention_mask'].to(device)), batch['labels'].to(device)
        else:
            return (batch['token_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['token_ids_2'].to(device), batch['attention_mask_2'].to(device)), batch['labels'].to(device)
    class Chore:
        def __init__(self, loss, dataloader, eval_fn, labels):
            self.loss = loss
            self.dataloader = dataloader
            self.eval_fn = eval_fn
            self.labels = labels

    sst_chore = Chore(F.cross_entropy, sst_train_dataloader, model.predict_sentiment, lambda batch: extract_labels(batch, True))
    sts_chore = Chore(lambda logits, b : F.mse_loss(logits.flatten().view(-1), b.float().flatten()), sts_train_dataloader, model.predict_similarity, lambda batch: extract_labels(batch, False))
    para_chore = Chore(lambda logits, b : F.binary_cross_entropy_with_logits(logits.flatten().view(-1), b.float().flatten()), para_train_dataloader, model.predict_paraphrase, lambda batch: extract_labels(batch, False))

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for i, chore in enumerate([sts_chore, sst_chore, para_chore]):
            for batch in tqdm(chore.dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                optimizer.zero_grad()
                
                eval_params, b_labels = chore.labels(batch)
                
                logits = chore.eval_fn(*eval_params)
                loss = chore.loss(logits, b_labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # print progress
                if num_batches % 100 == 0:
                    print(f"Epoch {epoch} batch {num_batches} train loss: {train_loss / num_batches}")
        
        train_loss /= num_batches
        print(f"Epoch {epoch} train loss: {train_loss}")
        model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        # Evaluate on dev set
        (paraphrase_accuracy, para_y_pred, para_sent_ids, 
                sentiment_accuracy,sst_y_pred, sst_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        # Save model if it's the best so far
        s = (paraphrase_accuracy + sentiment_accuracy + sts_corr) / 3
        if s > best_score:
            best_score = s
            print(f"Saving model with paraphrase accuracy {paraphrase_accuracy}")
            save_model(model, optimizer, args, config, args.filepath)

    # Load best model
    saved = torch.load(args.save_path)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    
    return model

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
