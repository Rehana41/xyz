import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import wordnet
import random
import nltk

nltk.download('wordnet')


def normalize_scores(df, target_min=0, target_max=10):
    scaler = MinMaxScaler(feature_range=(target_min, target_max))
    df['score'] = scaler.fit_transform(df[['score']])
    return df


def load_and_merge_essays(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[['essay', 'score']].dropna()
        df = normalize_scores(df)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def synonym_augment(text, p=0.1):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < p:
            synonyms = wordnet.synsets(word)
            if synonyms:
                lemmas = synonyms[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name().replace('_', ' ')
                    new_words.append(synonym)
                    continue
        new_words.append(word)
    return ' '.join(new_words)


def augment_dataset(df, augment_factor=0.3):
    augmented = []
    n_samples = int(len(df) * augment_factor)
    sampled_df = df.sample(n=n_samples, random_state=42)
    for _, row in sampled_df.iterrows():
        aug_essay = synonym_augment(row['essay'])
        augmented.append({'essay': aug_essay, 'score': row['score']})
    augmented_df = pd.DataFrame(augmented)
    return pd.concat([df, augmented_df], ignore_index=True)


def prepare_final_dataset(input_paths, output_path='data/essays.csv'):
    df = load_and_merge_essays(input_paths)
    df = augment_dataset(df, augment_factor=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Prepared and saved {len(df)} essays to {output_path}")


if __name__ == '__main__':
    input_paths = [
        'data/essay_set_1.csv',
        'data/essay_set_2.csv'
    ]
    prepare_final_dataset(input_paths)
