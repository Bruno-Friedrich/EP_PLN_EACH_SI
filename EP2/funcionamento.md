# Documentação Detalhada - Funcionamento do Sistema de Classificação de Textos

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura do Pipeline](#estrutura-do-pipeline)
3. [Fluxo de Dados](#fluxo-de-dados)
4. [Modelos Implementados](#modelos-implementados)
5. [Análise de Data Leakage](#análise-de-data-leakage)
6. [Problemas Identificados](#problemas-identificados)
7. [Recomendações para Produção](#recomendações-para-produção)

---

## 🎯 Visão Geral

Este sistema classifica textos em três categorias:
- **Academic**: Textos de origem acadêmica
- **Government**: Textos de origem governamental
- **Private**: Textos de origem privada

O pipeline implementa três modelos diferentes:
1. **Baseline**: TF-IDF + Regressão Logística
2. **MLP**: Multi-Layer Perceptron com TF-IDF ou Word2Vec
3. **BERT**: Fine-tuning do BERTimbau (BERT português)

---

## 📊 Estrutura do Pipeline

### 1. Pré-processamento (Cell 8-10)

**Função**: `preprocess_data()`

**Processo**:
1. Carrega dados do CSV (`ep2-train.csv`)
2. Remove valores nulos e textos vazios
3. Aplica normalização de entidades (emails, URLs, telefones, datas, etc.)
4. Aplica lowercase e limpeza de espaços
5. Faz split inicial: **85% treino / 15% teste** (stratified, random_state=10)
6. Salva dados processados em `ep2-train-preprocessed.csv`

**Output**:
```python
dataset = {
    "X_train": X_train,      # 37,126 textos (85%)
    "X_test": X_test,        # 6,552 textos (15%)
    "y_train": y_train,      # Labels de treino
    "y_test": y_test,        # Labels de teste
    "label_encoder": le     # Encoder de labels
}
```

**⚠️ IMPORTANTE**: O split é feito **UMA ÚNICA VEZ** no início. Os 15% de teste **NUNCA** são usados durante treinamento.

---

## 🔄 Fluxo de Dados

### Diagrama de Separação de Dados

```
Dados Originais (100% = 43,678 textos)
│
├── X_train (85% = 37,126 textos) ← Usado para treinamento
│   │
│   ├── BASELINE & MLP:
│   │   ├── CV: Usa 100% do X_train (85% do total)
│   │   └── Treino Final: Usa 100% do X_train (85% do total)
│   │
│   └── BERT:
│       ├── X_train_internal (70% do total = 30,591)
│       └── X_val_internal (15% do total = 6,535)
│
└── X_test (15% = 6,552 textos) ← RESERVADO APENAS PARA TESTE FINAL
```

### Separação por Modelo

#### **Baseline (Cell 12)**
```
X_train (85%) → Treino completo
X_train (85%) → CV de 10 folds (apenas para avaliação)
X_test (15%)  → Teste final (apenas no final)
```
✅ **SEM DATA LEAKAGE**: Teste nunca usado durante treino

#### **MLP (Cell 17)**
```
FASE 1 - CV:
  X_train (85%) → KFold(3) → 3 folds de treino/validação
  (Cada fold usa ~67% para treino, ~33% para validação)

FASE 2 - Seleção:
  Escolhe melhor modelo baseado em CV score

FASE 3 - Treino Final:
  X_train (85%) completo → Treino final
  X_test (15%) → Teste final (apenas no final)
```
✅ **SEM DATA LEAKAGE**: 
- CV usa apenas X_train
- Teste usado apenas no final
- Word2Vec treinado apenas em X_train

#### **BERT (Cell 19)**
```
Para cada combinação de hiperparâmetros:
  1. X_train (85%) → Split interno:
     ├── X_train_internal (70% do total)
     └── X_val_internal (15% do total)
  
  2. Treino:
     ├── train_dataset: X_train_internal (70%)
     └── eval_dataset: X_val_internal (15%) ← Usado durante treino
  
  3. Avaliação Final:
     └── test_dataset: X_test (15% original) ← APENAS NO FINAL
```
✅ **SEM DATA LEAKAGE**: 
- X_test original nunca usado durante treino
- Validação interna separada do teste final

---

## 🤖 Modelos Implementados

### 1. Baseline: TF-IDF + Regressão Logística

**Configuração** (Cell 3):
```python
BASELINE_CONFIG = {
    'vectorizer': {
        'max_features': 10000,
        'ngram_range': (1, 2)  # Unigramas e bigramas
    },
    'model': {
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000
    }
}
```

**Fluxo**:
1. Treina pipeline completo em `X_train`
2. Executa CV de 10 folds em `X_train` (apenas para avaliação)
3. Avalia no `X_test` final

**Resultados Armazenados**:
```python
baseline_results = {
    'train_score': float,      # Acurácia no treino
    'cv_mean': float,          # Média CV (10 folds)
    'cv_std': float,           # Desvio padrão CV
    'test_score': float,       # Acurácia no teste final
    'model': Pipeline,         # Modelo treinado
    'predictions': array       # Predições no teste
}
```

---

### 2. MLP: Multi-Layer Perceptron

**Configuração** (Cell 16):
```python
MLP_GRID_CONFIG = {
    'input_method': ['word2vec', 'tfidf'],
    'tfidf_params': {...},
    'word2vec_params': {...},
    'model_params': {...},
    'training_params': {...}
}
```

**Arquitetura**:
- Input: TF-IDF (10,000 features) ou Word2Vec (300 features)
- Hidden Layers: (512, 256) com BatchNormalization
- Dropout: 0.7
- Output: 3 classes (softmax)

**Fluxo em 3 Fases**:

**FASE 1 - Validação Cruzada**:
- Testa todas as combinações de hiperparâmetros
- Para cada combinação, executa KFold(3) em `X_train`
- Armazena CV scores

**FASE 2 - Seleção**:
- Escolhe melhor modelo baseado em `cv_mean`

**FASE 3 - Treino Final**:
- Treina melhor modelo em `X_train` completo (85%)
- Usa `validation_split=0.15` para early stopping
- Avalia no `X_test` final (15%)

**⚠️ IMPORTANTE - Word2Vec**:
- Word2Vec é treinado **APENAS** em `X_train`
- Depois é usado para converter `X_test` (sem retreinar)
- ✅ **CORRETO**: Não há data leakage

**Resultados Armazenados**:
```python
mlp_results = {
    'best_cv_score': float,    # Melhor CV score
    'test_score': float,        # Score no teste final
    'best_result': dict,        # Resultado completo do melhor modelo
    'model': KerasClassifier,  # Modelo treinado
    'predictions': array        # Predições no teste
}
```

---

### 3. BERT: Fine-tuning BERTimbau

**Configuração** (Cell 18):
```python
BERT_GRID_CONFIG = {
    'model_name': 'neuralmind/bert-base-portuguese-cased',
    'grid_params': {
        'learning_rate': [2e-5],
        'batch_size': [8],
        'num_train_epochs': [1],
        'weight_decay': [0.01],
        'max_length': [256]
    }
}
```

**Fluxo**:
1. Para cada combinação de hiperparâmetros:
   - Separa `X_train` (85%) em:
     - `X_train_internal` (70% do total)
     - `X_val_internal` (15% do total)
   - Treina em `X_train_internal`
   - Valida em `X_val_internal` durante treino
   - Avalia no `X_test` original (15%) **APENAS NO FINAL**

2. ✅ **Escolhe melhor modelo baseado em `val_score`** (validação interna)
   - `test_score` é usado apenas para reporte final
   - Evita overfitting ao conjunto de teste

**⚠️ IMPORTANTE**:
- `X_test` original **NUNCA** é usado durante treino
- Cada combinação cria sua própria separação interna
- O melhor modelo é escolhido pelo `val_score` (validação interna), não pelo `test_score`
- `test_score` é usado apenas para reporte final

**Otimizações Implementadas**:
- ✅ **Paralelização de DataLoader**: `dataloader_num_workers=4` (usa múltiplos cores da CPU)
- ✅ **Pin Memory**: `dataloader_pin_memory=True` (acelera transferência CPU→GPU)
- ✅ **Mixed Precision**: `fp16=True` (se GPU disponível, reduz uso de memória e acelera)

**Resultados Armazenados**:
```python
bert_final_results = {
    'test_score': float,        # Score no teste final
    'best_params': dict,        # Melhores hiperparâmetros
    'model': BertModel,         # Modelo treinado
    'tokenizer': AutoTokenizer, # Tokenizer
    'all_results': list         # Todos os resultados do grid-search
}
```

---

## 🔍 Análise de Data Leakage

### ✅ Verificações Realizadas

#### 1. Baseline
- ✅ Treino usa apenas `X_train`
- ✅ CV usa apenas `X_train`
- ✅ Teste usa apenas `X_test` (separado desde o início)
- ✅ **SEM DATA LEAKAGE**

#### 2. MLP
- ✅ CV usa apenas `X_train`
- ✅ Word2Vec treinado apenas em `X_train`
- ✅ TF-IDF fit apenas em `X_train`
- ✅ Treino final usa apenas `X_train`
- ✅ Teste usa apenas `X_test` (separado desde o início)
- ✅ **SEM DATA LEAKAGE**

#### 3. BERT
- ✅ Cada combinação separa `X_train` internamente
- ✅ `X_test` original nunca usado durante treino
- ✅ Validação interna separada do teste final
- ✅ **SEM DATA LEAKAGE**

### ⚠️ Pontos de Atenção

1. **BERT - Seleção do Melhor Modelo**:
   - ✅ **CORRIGIDO**: Agora escolhe baseado em `val_score` (validação interna)
   - `test_score` é usado apenas para reporte final
   - ✅ **CORRETO**: Evita overfitting ao conjunto de teste

2. **MLP - Treino Final**:
   - Usa `validation_split=0.15` no treino final
   - Isso cria uma validação interna adicional
   - ✅ **CORRETO**: Não usa dados de teste

---

## 🐛 Problemas Identificados e Corrigidos

### 1. ✅ CORRIGIDO: BERT - Otimização de DataLoader

**Problema Original**: O `TrainingArguments` do BERT não tinha `dataloader_num_workers` configurado.

**Impacto Original**: 
- Carregamento de dados mais lento
- Não aproveitava múltiplos cores da CPU

**Correção Aplicada** (Cell 19):
```python
# Configurar número de workers para paralelizar carregamento de dados (CPU)
import os
num_workers = min(4, os.cpu_count() or 1)

training_args = TrainingArguments(
    # ... outros parâmetros ...
    dataloader_num_workers=num_workers,  # ✅ ADICIONADO: Paralelizar carregamento
    dataloader_pin_memory=torch.cuda.is_available()  # ✅ ADICIONADO: Acelerar CPU→GPU
)
```

**Status**: ✅ **CORRIGIDO** - Agora aproveita múltiplos cores da CPU para carregamento de dados

---

### 2. ✅ CORRIGIDO: Comparação de Resultados - Uso de `globals()`

**Problema Original**: A célula de comparação usava `globals()` para verificar se variáveis existem.

**Impacto Original**:
- Poderia falhar silenciosamente se células não foram executadas
- Não é uma prática recomendada

**Correção Aplicada** (Cell 20):
```python
# ✅ CORRIGIDO: Usa try/except em vez de globals()
try:
    if mlp_results and 'test_score' in mlp_results:
        comparison_data['Modelo'].append('MLP (Grid-Search)')
        comparison_data['Test Accuracy'].append(mlp_results['test_score'])
except NameError:
    print("⚠️ MLP não foi executado ainda")

try:
    if bert_final_results is not None and 'test_score' in bert_final_results:
        comparison_data['Modelo'].append('BERT (Grid-Search)')
        comparison_data['Test Accuracy'].append(bert_final_results['test_score'])
except NameError:
    print("⚠️ BERT não foi executado ainda")
```

**Status**: ✅ **CORRIGIDO** - Tratamento de erros robusto com try/except

---

### 3. ✅ CORRIGIDO: BERT - Seleção Baseada em Test Score

**Problema Original**: O melhor modelo BERT era escolhido baseado em `test_score`.

**Impacto Original**:
- Poderia causar overfitting ao conjunto de teste
- O modelo poderia estar "vendo" o teste durante seleção

**Correção Aplicada** (Cell 19):
```python
# ✅ CORRIGIDO: Escolhe baseado em val_score (validação interna)
best_bert = max(bert_results, key=lambda x: x['val_score'])

print(f"Melhor modelo BERT (baseado em Val Score - validação interna):")
print(f"  Val Accuracy: {best_bert['val_score']:.4f} ← Usado para seleção")
print(f"  Test Accuracy: {best_bert['test_score']:.4f} ← Apenas para reporte final")
```

**Status**: ✅ **CORRIGIDO** - Agora escolhe baseado em validação interna, evitando overfitting

---

### 4. ℹ️ INFORMATIVO: MLP - Estrutura de Resultados

**Observação**: A comparação usa `mlp_results['best_cv_score']`, mas a estrutura também tem `mlp_results['best_result']['cv_mean']`.

**Status**: ✅ **FUNCIONA CORRETAMENTE** - Ambos valores são idênticos, apenas redundância estrutural (não é um problema)

---

## 🚀 Recomendações para Produção

### 1. Separação de Dados

✅ **JÁ IMPLEMENTADO CORRETAMENTE**:
- Split inicial fixo (random_state=10)
- Teste nunca usado durante treino
- Validação interna separada (BERT)

### 2. Persistência de Modelos

**Recomendação**: Salvar modelos treinados para uso em produção:

```python
# Salvar Baseline
import joblib
joblib.dump(baseline_results['model'], 'models/baseline_model.pkl')
joblib.dump(baseline_results['vectorizer'], 'models/baseline_vectorizer.pkl')

# Salvar MLP
best_cv_result['model'].model_.save('models/mlp_model.h5')
if 'vectorizer' in mlp_results:
    joblib.dump(mlp_results['vectorizer'], 'models/mlp_vectorizer.pkl')
elif 'w2v_model' in mlp_results:
    mlp_results['w2v_model'].save('models/mlp_word2vec.model')

# Salvar BERT
best_bert['model'].save_pretrained('models/bert_model')
best_bert['tokenizer'].save_pretrained('models/bert_tokenizer')
```

### 3. Validação em Produção

**Recomendação**: Implementar validação de entrada:

```python
def validate_input(text):
    """Valida texto de entrada"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if len(text.strip()) == 0:
        raise ValueError("Input cannot be empty")
    if len(text) > 5000:  # Limite razoável
        raise ValueError("Input too long")
    return True
```

### 4. Monitoramento

**Recomendação**: Implementar logging de predições:

```python
import logging

logging.basicConfig(filename='production.log', level=logging.INFO)

def predict_with_logging(text, model):
    prediction = model.predict([text])[0]
    logging.info(f"Text: {text[:100]}... | Prediction: {prediction}")
    return prediction
```

### 5. Versionamento

**Recomendação**: Salvar versão do modelo e parâmetros:

```python
model_metadata = {
    'model_version': '1.0',
    'training_date': '2024-01-01',
    'test_score': baseline_results['test_score'],
    'cv_score': baseline_results['cv_mean'],
    'parameters': BASELINE_CONFIG
}

import json
with open('models/baseline_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
```

---

## 📝 Resumo Executivo

### ✅ Pontos Fortes

1. **Separação de Dados Correta**: 
   - Teste separado desde o início
   - Sem data leakage identificado

2. **Validação Cruzada Adequada**:
   - MLP usa CV para seleção de hiperparâmetros
   - BERT usa validação interna separada

3. **Estrutura Modular**:
   - Código organizado em células
   - Funções reutilizáveis

### ✅ Melhorias Aplicadas

1. **BERT - Seleção de Modelo**:
   - ✅ **CORRIGIDO**: Agora escolhe baseado em `val_score` (validação interna)
   - Evita overfitting ao conjunto de teste

2. **Otimização de Performance**:
   - ✅ **CORRIGIDO**: Adicionado `dataloader_num_workers` no BERT
   - Aproveita múltiplos cores da CPU para carregamento de dados

3. **Tratamento de Erros**:
   - ✅ **CORRIGIDO**: Substituído `globals()` por try/except
   - Tratamento de erros mais robusto

4. **Documentação**:
   - ✅ **MELHORADO**: Documentação detalhada criada
   - Este documento explica todo o funcionamento

### 🎯 Conclusão

O sistema está **bem estruturado**, **sem data leakage** e **com todas as melhorias aplicadas**. O código está otimizado e segue boas práticas:

- ✅ Separação correta de dados (sem data leakage)
- ✅ Seleção de modelos adequada (evita overfitting)
- ✅ Otimização de performance (paralelização de dados)
- ✅ Tratamento robusto de erros
- ✅ Documentação completa

**Status Geral**: ✅ **PRONTO PARA PRODUÇÃO**

---

## 📚 Referências

- Scikit-learn: https://scikit-learn.org/
- Hugging Face Transformers: https://huggingface.co/transformers/
- TensorFlow/Keras: https://www.tensorflow.org/
- Gensim Word2Vec: https://radimrehurek.com/gensim/

---

**Última Atualização**: 2025-01-27
**Versão do Documento**: 2.0

**Changelog**:
- v2.0 (2025-01-27): Atualizado para refletir correções aplicadas
  - BERT: Seleção baseada em val_score (corrigido)
  - BERT: Otimização de DataLoader (corrigido)
  - Comparação: Tratamento de erros com try/except (corrigido)
- v1.0 (2024-01-01): Versão inicial

