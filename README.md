# Maturitaetsarbeit
Maturitätsarbeit Gian Seeholzer
Wie bereits in der Methodik beschrieben, werden wird zuerst ein LSTM auf ausgeglichenen Daten trainieren und testen und dieses dann auf unausgeglichenen Daten weiter auf die Probe stellen.
Darauf werden wir ein BERT Modell erstellen und dessen Sentiment Analyse von YouTube Kommentaren und den Vorhersagen des ersten Modells nutzen, um ein neues Datenset zu schaffen mit den errechneten Wahrscheinlichkeiten für die Klassenzugehörigkeiten und dem YouTube-Kommentar Sentiment

Warum der Code so viele Quellen hat: Da ich zu Beginn meiner Maturitätsarbeit das meiste von dem nicht konnte, habe ich viel durch Learning by doing gelernt. Zum einen war die Quellenlage zum Lernen für solch komplexer Modelle spärlich und zum anderen hielt ich es für unmöglich den Ort nicht als Quelle anzugeben, von welchem ich etwas gelernt hatte.

Häufig verwendete Quellen, die zu umständlich wären um überall aufzuführen:
Der gesamte folgende Abschnitt und alle weiteren LSTMs stammt von zu grössten Teilen von Venelin Valkov, welcher eine hervorragende Serie zu diesem Thema erstellt hat. Die Videos sind unter den folgenden Links zu finden:
https://www.youtube.com/watch?v=jR0phoeXjrc&ab_channel=VenelinValkov
https://www.youtube.com/watch?v=PCgrgHgy26c&ab_channel=VenelinValkov

rows = [dict(sentiment=row.sentiment, pred1=row.pred1, pred2=row.pred2, label=row.label) for _, row in df.iterrows()]
features_df = pd.DataFrame(rows)
features_df['label'] = features_df['label'].shift(1)
features_df = features_df.dropna()
print(features_df)
#-------------------------------------------------------------------------------------------------------------
len_feat_df = len(features_df)
train_df, test_df = features_df[:-130], features_df[-130:]
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):
    sequences = []
    data_size = len(input_data)
    for i in range(data_size - sequence_length):
        sequence = input_data[i:i+sequence_length]
        label_position = i  + sequence_length
        label = input_data.iloc[label_position][target_column]
        sequences.append((sequence, label))
    return sequences

SEQUENCE_LENGTH = 30
train_sequences = create_sequences(train_df, "label", SEQUENCE_LENGTH)
test_sequences = create_sequences(test_df, "label", SEQUENCE_LENGTH)
#-------------------------------------------------------------------------------------------------------------
class APPLDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(sequence=torch.Tensor(sequence.to_numpy()), label=torch.tensor(label).long())
class APPLPriceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_dataset = APPLDataset(self.train_sequences)
        self.test_dataset = APPLDataset(self.test_sequences)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
    #def val_dataloader(self):
        #return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)
#-------------------------------------------------------------------------------------------------------------
N_EPOCHS = 200
BATCH_SIZE = 64
n_features=4
data_module = APPLPriceDataModule(train_sequences, test_sequences, BATCH_SIZE)
data_module.setup()
#-------------------------------------------------------------------------------------------------------------
class PricePredictionModel(nn.Module):
    def __init__(self, features, n_classes, n_hidden=256, n_layers=3):
        super().__init__()
        #self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, batch_first=True, num_layers=n_layers, dropout=0.8)  
        self.classifier = nn.Linear(n_hidden, n_classes)
    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)
#-------------------------------------------------------------------------------------------------------------
class APPLPricePredictor(pl.LightningModule):
    def __init__(self, n_features=4, n_classes=2):
        super().__init__()
        self.model = PricePredictionModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    def training_step(self, batch ,batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        return {"loss": loss, "accuracy": step_accuracy}
    def test_step(self, batch ,batch_idx):
        print("happens")
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        return {"loss": loss, "accuracy": step_accuracy}
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.0001)
#-------------------------------------------------------------------------------------------------------------
that_model = APPLPricePredictor(n_features=train_df.shape[1], n_classes=2)
#-------------------------------------------------------------------------------------------------------------
trainer = pl.Trainer(max_epochs=N_EPOCHS)
trainer.fit(that_model, data_module)


Ebenfalls nicht vor mir ist:
data = ticker.history(period='10y',
                      interval='1d',
                      start=None,
                      end=None,
                      actions=True,
                      auto_adjust=True,
                      back_adjust=False)
Dieser Code stammt aus dem Github der  zweiten Auflage von Machine Learning for Algorithmic Trading von Stefan Jansen, herausgegeben 2020 von Packt Publishing. Den Link finden Sie unter: https://github.com/fadhliong/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/02_market_and_fundamental_data/03_data_providers/02_yfinance_demo.ipynb
