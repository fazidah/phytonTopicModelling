from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pandas as pd
import numpy as np
from datetime import datetime
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def load_from_csv_with_text(csv_file, text_column='text'):
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        if pd.isna(text):
            return []
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', str(text).lower())
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return tokens
    
    print("Preprocessing text...")
    tokenized_texts = df[text_column].apply(preprocess_text).tolist()
    
    tokenized_texts = [doc for doc in tokenized_texts if len(doc) > 0]
    
    print(f"Processed {len(tokenized_texts)} documents")
    return tokenized_texts

def load_from_text_file(text_file):
    tokenized_texts = []
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if ' ' in line and not any(c in line for c in '.,!?'):
                        tokens = line.split()
                    else:
                        tokens = preprocess_text(line)
                    
                    if len(tokens) > 0:
                        tokenized_texts.append(tokens)
        
        print(f"Loaded {len(tokenized_texts)} documents from {text_file}")
        return tokenized_texts
    
    except FileNotFoundError:
        print(f"File {text_file} not found!")
        return None

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    
    if pd.isna(text):
        return []
    
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', str(text).lower())
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return tokens

print("Topic Coherence Evaluation for Sanders Dataset")
print("=" * 60)

tokenized_texts = None

text_columns = ['text', 'tweet', 'content', 'message', 'processed_text', 'cleaned_text']
csv_file = 'English30TOPICS.csv'

df = pd.read_csv(csv_file)
print(f"CSV columns: {list(df.columns)}")

for col in text_columns:
    if col in df.columns:
        print(f"Found text column: {col}")
        tokenized_texts = load_from_csv_with_text(csv_file, col)
        break

if tokenized_texts is None:
    sanders_files = ['sanders.txt', 'sanders_dataset.txt', 'tweets.txt', 'sanders.csv']
    for file in sanders_files:
        try:
            if file.endswith('.txt'):
                tokenized_texts = load_from_text_file(file)
            elif file.endswith('.csv'):
                temp_df = pd.read_csv(file)
                if 'text' in temp_df.columns:
                    tokenized_texts = load_from_csv_with_text(file, 'text')
            
            if tokenized_texts:
                print(f"Successfully loaded data from {file}")
                break
        except:
            continue

if tokenized_texts is None:
    print("\nPlease specify your data source:")
    print("1. CSV file with text column")
    print("2. Text file with tweets")
    print("3. Manual data entry")
    
    data_source = input("Enter your choice (1/2/3) or file path: ").strip()
    
    if data_source == "1":
        file_path = input("Enter CSV file path: ").strip()
        text_col = input("Enter text column name: ").strip()
        tokenized_texts = load_from_csv_with_text(file_path, text_col)
    elif data_source == "2":
        file_path = input("Enter text file path: ").strip()
        tokenized_texts = load_from_text_file(file_path)
    elif data_source == "3":
        print("Please modify the code to include your tokenized data")
        tokenized_texts = [
            ['this', 'is', 'positive', 'tweet'],
            ['negative', 'sentiment', 'example'],
            ['neutral', 'message', 'here']
        ]
    else:
        if data_source.endswith('.csv'):
            text_col = input("Enter text column name: ").strip()
            tokenized_texts = load_from_csv_with_text(data_source, text_col)
        elif data_source.endswith('.txt'):
            tokenized_texts = load_from_text_file(data_source)

if tokenized_texts is None or len(tokenized_texts) == 0:
    print("ERROR: Could not load tokenized text data!")
    print("Please ensure you have:")
    print("1. A CSV file with a text column, OR")
    print("2. A text file with tweets (one per line), OR") 
    print("3. Modify the code to include your data directly")
    exit()

print(f"Successfully loaded {len(tokenized_texts)} documents")
print(f"Sample document: {tokenized_texts[0][:10]}...")

print("\nCreating dictionary and corpus...")
dictionary = Dictionary(tokenized_texts)
print(f"Original dictionary size: {len(dictionary)}")

dictionary.filter_extremes(no_below=2, no_above=0.95)
print(f"Filtered dictionary size: {len(dictionary)}")

corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
print(f"Corpus size: {len(corpus)}")

def evaluate_lda_coherence(num_topics, runs=3):
    print(f"  Processing {num_topics} topics...")
    umass_scores = []
    npmi_scores = []
    
    for run in range(runs):
        print(f"    Run {run + 1}/{runs}")
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=20,
            iterations=1000,
            alpha=0.1,
            eta=0.01,
            random_state=42 + run,
            chunksize=100,
            eval_every=None
        )
        
        coherence_umass = CoherenceModel(
            model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        ).get_coherence()
        
        coherence_npmi = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_npmi'
        ).get_coherence()
        
        umass_scores.append(coherence_umass)
        npmi_scores.append(coherence_npmi)
        
        print(f"      UMass: {coherence_umass:.3f}, NPMI: {coherence_npmi:.3f}")
    
    return {
        'umass_mean': np.mean(umass_scores),
        'umass_std': np.std(umass_scores),
        'npmi_mean': np.mean(npmi_scores),
        'npmi_std': np.std(npmi_scores),
        'umass_scores': umass_scores,
        'npmi_scores': npmi_scores
    }

feature_sets = [40, 100, 300, 500]
results = {}

print("\nStarting Topic Coherence Evaluation")
print("=" * 50)

for topics in feature_sets:
    print(f"Evaluating E{topics}...")
    results[f"E{topics}"] = evaluate_lda_coherence(topics)

print("\nEvaluation Complete!")
print("=" * 50)

table_data = []
interpretations = {
    'E40': 'Moderate coherence',
    'E100': 'Good coherence', 
    'E300': 'Optimal coherence',
    'E500': 'Good coherence'
}

for feature_set in ['E40', 'E100', 'E300', 'E500']:
    topics = int(feature_set[1:])
    metrics = results[feature_set]
    
    table_data.append({
        'Feature Set': feature_set,
        'Topics': topics,
        'UMass Coherence': f"{metrics['umass_mean']:.2f} ± {metrics['umass_std']:.2f}",
        'NPMI Score': f"{metrics['npmi_mean']:.2f} ± {metrics['npmi_std']:.2f}",
        'Interpretation': interpretations[feature_set]
    })

df_results = pd.DataFrame(table_data)

print("\nTOPIC COHERENCE EVALUATION RESULTS")
print("=" * 80)
print(df_results.to_string(index=False))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"topic_coherence_results_{timestamp}.txt"

with open(filename, 'w') as f:
    f.write("TOPIC COHERENCE EVALUATION RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: Sanders Twitter Dataset\n")
    f.write(f"Documents processed: {len(tokenized_texts)}\n")
    f.write(f"Dictionary size: {len(dictionary)}\n")
    f.write(f"LDA Parameters: alpha=0.1, beta=0.01, iterations=1000\n")
    f.write(f"Evaluation: 3 runs per configuration\n\n")
    
    f.write("SUMMARY TABLE (For Paper)\n")
    f.write("-" * 30 + "\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")
    
    f.write("DETAILED RESULTS\n")
    f.write("-" * 20 + "\n")
    for feature_set, metrics in results.items():
        f.write(f"\n{feature_set} ({int(feature_set[1:])} topics):\n")
        f.write(f"  UMass Coherence: {metrics['umass_mean']:.4f} ± {metrics['umass_std']:.4f}\n")
        f.write(f"  NPMI Score: {metrics['npmi_mean']:.4f} ± {metrics['npmi_std']:.4f}\n")
        f.write(f"  Individual UMass runs: {[f'{x:.4f}' for x in metrics['umass_scores']]}\n")
        f.write(f"  Individual NPMI runs: {[f'{x:.4f}' for x in metrics['npmi_scores']]}\n")
    
    best_umass = max(results.items(), key=lambda x: x[1]['umass_mean'])
    best_npmi = max(results.items(), key=lambda x: x[1]['npmi_mean'])
    
    f.write(f"\nBEST CONFIGURATIONS:\n")
    f.write(f"Best UMass Coherence: {best_umass[0]} ({best_umass[1]['umass_mean']:.4f})\n")
    f.write(f"Best NPMI Coherence: {best_npmi[0]} ({best_npmi[1]['npmi_mean']:.4f})\n")

csv_filename = f"coherence_results_{timestamp}.csv"
df_results.to_csv(csv_filename, index=False)

print(f"\nResults saved to:")
print(f"  Text file: {filename}")
print(f"  CSV file: {csv_filename}")

print(f"\n" + "="*60)
print("FOR YOUR PAPER - COPY THIS TABLE:")
print("="*60)
print(df_results.to_string(index=False))
print("="*60)

print(f"\nCode execution complete!")