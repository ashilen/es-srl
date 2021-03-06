import json
import argparse
from typing import Dict, Iterable

import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.tokenizers import Token
from allennlp.data.fields import IndexField, TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, Seq2VecEncoder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.util import evaluate


@DatasetReader.register('classification-uds')
class UDSReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, 'r') as file:
            data = json.load(file)

            for example in data:
                tokens = [Token(token) for token in example["tokens"].split()]

                pred_idx, pred_str = example["pred_head"]
                arg_idx, arg_str = example["arg_head"]

                # Tack on the arg's deprel to its token representation.
                tokens[arg_idx] = Token(arg_str[0], dep_=example["deprel"])

                tokens_field = TextField(tokens, self.token_indexers)
                pred_head_field = IndexField(pred_idx, tokens_field)
                arg_head_field = IndexField(arg_idx, tokens_field)
                label_field = LabelField(str(example["label"]))

                fields = {
                    "tokens": tokens_field,
                    "label": label_field,
                    "pred_head": pred_head_field,
                    "arg_head": arg_head_field,
                }

                yield Instance(fields)


@Model.register('simple_classifier')
class UDSModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(positive_label=1)

    def forward(self,
                tokens: TextFieldTensors,
                pred_head: torch.LongTensor,
                arg_head: torch.LongTensor,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self.accuracy(logits, label)
        self.f1(logits, label)
        return {'loss': loss, 'probs': probs}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.f1.get_metric(reset)


def ModelFactory(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return UDSModel(vocab, embedder, encoder)


def TrainerFactory(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


def data_files(protorole):
    base = "data/%s/" % protorole

    return (
        "%s/train" % base,
        "%s/dev" % base,
        "%s/min_pair" % base,
        "%s/serialize" % base,
    )


def train_and_test_model(protorole):
    (
        train_file, dev_file,
        test_file, serialization_dir
    ) = data_files(protorole)

    reader = UDSReader()

    train_data = list(reader.read(train_file))
    dev_data = list(reader.read(dev_file))

    vocab = Vocabulary.from_instances(train_data + dev_data)
    model = ModelFactory(vocab)

    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)

    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    trainer = TrainerFactory(model, serialization_dir, train_loader, dev_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")

    # Now we can evaluate the model on a new dataset.
    test_data = list(reader.read(test_file))
    data_loader = SimpleDataLoader(test_data, batch_size=8)
    data_loader.index_with(model.vocab)

    results = evaluate(model, data_loader)
    print(results)


ROLES = ["agent", "patient", "theme",
         "experiencer", "destination"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model the UDS.")
    parser.add_argument("--protorole", type=str,
                        choices=ROLES,
                        help="Protorole to model.")

    args = parser.parse_args()

    if args.protorole:
        train_and_test_model(args.protorole)
    else:
        for role in ROLES:
            print(role.upper())
            train_and_test_model(role)
            print("\n")
