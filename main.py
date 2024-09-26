import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Cargar el dataset desde un CSV
data = pd.read_csv("dataset_compras.csv")

# Crear un objeto Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Ver un ejemplo de los datos
print(dataset[0])

# Tokenizador del modelo BERT preentrenado
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Función para tokenizar el dataset
def tokenize_function(examples):
    return tokenizer(examples["descripcion"], padding="max_length", truncation=True)

# Aplicar la tokenización al dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Ver los datos tokenizados
print(tokenized_dataset[0])



from sklearn.model_selection import train_test_split

# Dividir el dataset en entrenamiento y validación
train_test_data = data[['descripcion', 'categoria']]
train_data, test_data = train_test_split(train_test_data, test_size=0.2)

# Convertir de nuevo en Dataset de Hugging Face
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Crear etiquetas numéricas
labels = train_data['categoria'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Añadir etiquetas numéricas al dataset
def encode_labels(example):
    example['label'] = label2id[example['categoria']]
    return example

train_dataset = train_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

# Eliminar la columna original de categorías
train_dataset = train_dataset.remove_columns("categoria")
test_dataset = test_dataset.remove_columns("categoria")



from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Cargar el modelo preentrenado
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Entrenar el modelo
trainer.train()



# Evaluar el modelo
results = trainer.evaluate()

# Mostrar los resultados
print(results)



# Guardar el modelo entrenado
trainer.save_model("./modelo_clasificacion_compras")
tokenizer.save_pretrained("./modelo_clasificacion_compras")



# Cargar el modelo y el tokenizador entrenado
from transformers import pipeline

classifier = pipeline("text-classification", model="./modelo_clasificacion_compras", tokenizer="./modelo_clasificacion_compras")

# Probar con un nuevo ejemplo
resultado = classifier("Compré una camisa nueva de algodón")
print(resultado)
