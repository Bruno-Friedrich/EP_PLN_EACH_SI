# Classificação de Estilos Textuais com PLN - Versículos Bíblicos

Repositório desenvolvido para o **Exercício Programa (EP)** da disciplina de **Processamento de Língua Natural (PLN)** - EACH USP.

Este projeto implementa um pipeline completo de Machine Learning para classificação de textos em três diferentes pares de estilos literários: **Arcaico vs Moderno**, **Complexo vs Simples**, e **Literal vs Dinâmico**.

---

## **Sumário**

- [Visão Geral](#-visão-geral)
- [Datasets](#-datasets)
- [Metodologia](#-metodologia)
- [Pipeline de Processamento](#-pipeline-de-processamento)
- [Modelos Implementados](#-modelos-implementados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)

---

## **Visão Geral**

O projeto visa desenvolver modelos de classificação de textos capazes de distinguir diferentes estilos literários. Utilizamos técnicas de **Processamento de Língua Natural (PLN)** e **Machine Learning** para treinar, validar e testar modelos em três tarefas distintas de classificação binária.

**Principais características:**

- Pipeline completo (pré-processamento → vetorização → treinamento → validação → teste)
- Grid Search com otimização de hiperparâmetros (TF-IDF + Modelos)
- Validação Cruzada (10-fold CV) para avaliação robusta
- Hold-out set (15%) para teste final imparcial
- Comparação sistemática entre 3 modelos e 3 datasets

---

## **Datasets**

O projeto trabalha com três conjuntos de dados balanceados:

| Dataset                        | Classes            | Total de Linhas | Proporção | Balanceamento            |
| ------------------------------ | ------------------ | --------------- | --------- | ------------------------ |
| **train_arcaico_moderno.csv**  | arcaico / moderno  | 36,884          | 50% / 50% | Perfeitamente balanceado |
| **train_complexo_simples.csv** | complexo / simples | 33,422          | 50% / 50% | Perfeitamente balanceado |
| **train_literal_dinamico.csv** | literal / dinâmico | 36,964          | 50% / 50% | Perfeitamente balanceado |

**Estrutura dos CSVs:**

```
text;style
"Texto de exemplo aqui...";arcaico
"Outro texto exemplo...";moderno
```

**Análise de qualidade:**

- Todos os datasets são balanceados (razão 1.00x)
- Nenhum valor nulo encontrado (exceto 1 em train_complexo_simples.csv)
- Textos em português (variados tamanhos e complexidades)

---

## **Metodologia**

### **1. Pré-Processamento**

Aplicamos técnicas minimalistas de pré-processamento para preservar características estilísticas importantes:

```python
preprocess_params = {
    "lowercase": True,                  # Conversão para minúsculas
    "normalize_unicode": False,         # Mantém caracteres especiais
    "remove_extra_whitespace": True,    # Remove espaços extras
    "remove_punct": False,              # Mantém pontuação (importante para estilo)
}
```

### **2. Separação Treino/Teste**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,      # 85% treino / 15% teste
    stratify=y,          # Mantém proporção das classes
    random_state=10      # Reprodutibilidade
)
```

- **Stratified split:** Mantém proporção 50/50 em treino e teste
- **Hold-out set:** 15% dos dados nunca são vistos durante treino/validação
- **Reprodutibilidade:** `random_state=10` garante resultados consistentes

obs: `random_state=42` é usado nos modelos, e no embaralhamento do pré-processamento.

### **3. Vetorização (TF-IDF)**

Usamos **TF-IDF (Term Frequency-Inverse Document Frequency)** integrado ao pipeline via `GridSearchCV`:

**Hiperparâmetros otimizados:**

```python
'vectorizer__max_features': [3000, 5000, 10000],  # Número de features
'vectorizer__ngram_range': [(1, 1), (1, 2)],      # Unigrams e Bigrams
'vectorizer__min_df': [2, 5],                     # Frequência mínima
'vectorizer__max_df': [0.9, 0.95],                # Frequência máxima (remove stop words)
```

### **4. Pipeline de Treinamento**

Utilizamos `sklearn.pipeline.Pipeline` para integrar vetorização e modelo:

```python
Pipeline([
    ('vectorizer', TfidfVectorizer()),  # Etapa 1: Texto → Vetores
    ('model', MultinomialNB())          # Etapa 2: Vetores → Classificação
])
```

**Validação Cruzada (10-fold CV)**

```python
GridSearchCV(
    pipeline,
    param_grid,
    cv=10,              # 10 folds
    scoring='accuracy',
    n_jobs=-1           # Paralelização
)
```

**Como funciona:**

```
Dados de Treino (85%)
    ↓
Dividir em 10 partes (folds)
    ↓
Para cada combinação de hiperparâmetros:
    Fold 1: Treina em 9/10, valida em 1/10
    Fold 2: Treina em 9/10, valida em 1/10
    ...
    Fold 10: Treina em 9/10, valida em 1/10
    ↓
Média das 10 acurácias = Score do CV
    ↓
Escolhe a melhor combinação
```

**Vantagens:**

- Mais robusto que train/val split simples
- Detecta overfitting

### **6. Teste Final (Hold-out Set)**

- Dados nunca vistos durante treino/validação

```python
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

---

## **Modelos Implementados**

### **1. Naive Bayes (MultinomialNB)**

```python
MultinomialNB(alpha=...)
```

**Hiperparâmetros otimizados:**

- `alpha`: [0.1, 0.5, 1.0, 2.0] (suavização de Laplace)

### **2. Logistic Regression**

```python
LogisticRegression(C=..., solver=..., class_weight=...)
```

**Hiperparâmetros otimizados:**

- `C`: [0.1, 1.0, 10.0] (regularização inversa)
- `solver`: ['lbfgs', 'liblinear']
- `class_weight`: ['balanced', None]

### **3. SVM (LinearSVC)**

```python
LinearSVC(C=..., max_iter=..., dual=False)
```

**Hiperparâmetros otimizados:**

- `C`: [0.1, 1.0, 10.0] (regularização)
- `max_iter`: [1000, 2000]

## **Estrutura do Projeto**

```
EP_PLN_EACH_SI/
│
├── treino/                                 # Datasets de treinamento
│   ├── train_arcaico_moderno.csv          # Dataset 1
│   ├── train_complexo_simples.csv         # Dataset 2
│   └── train_literal_dinamico.csv         # Dataset 3
│
├── EP.ipynb                                # Notebook principal
├── README.md                               # Este arquivo
```

---

## **Como Executar**

### **Pré-requisitos**

- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Bibliotecas listadas em [Requisitos](#-requisitos)

### **Passo a Passo**

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/EP_PLN_EACH_SI.git
cd EP_PLN_EACH_SI
```

2. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

3. **Abra o Jupyter Notebook:**

```bash
jupyter notebook EP.ipynb
```

4. **Execute todas as células:**

- No menu: `Cell` → `Run All`
- Ou: `Ctrl+Shift+Enter` em cada célula

5. **Visualize os resultados:**

- Análise de balanceamento dos datasets
- Resultados do Grid Search (CV)
- Teste final no hold-out set
- Comparação entre datasets

## **Requisitos**

### **Bibliotecas Python**

```python
# Manipulação de dados
pandas>=1.3.0
numpy>=1.21.0

# Machine Learning
scikit-learn>=1.0.0

# Jupyter
jupyter>=1.0.0
notebook>=6.4.0
```

### **Instalar dependências:**

```bash
pip install pandas numpy scikit-learn jupyter notebook
```

---

## **Autores**

- Brunão Friedrich
- Lucas Ferracin
- Pablo Assunção
- Yannis Pontuschka
