# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : _Davoust, Kilian_
- **Projet** : _IMDb (analyse de sentiments binaire) avec BiLSTM et attention_
- **Dépôt Git** : _[URL publique](https://huggingface.co/datasets/stanfordnlp/imdb)_
- **Environnement** : `python == 3.12.3`, `torch == 2.8.0`, `cuda == None`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) : https://huggingface.co/datasets/stanfordnlp/imdb
- **Type d’entrée** (image / texte / audio / séries) : Texte
- **Tâche** (multiclasses, multi-label, régression) : Classification binaire (sentiment positif/négatif)
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : (256,)
- **Nombre de classes** (`meta["num_classes"]`) : 1

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

Dataset IMDb (Large Movie Review Dataset) provenant de https://huggingface.co/datasets/stanfordnlp/imdb.
Contient 50,000 critiques de films en anglais pour analyse de sentiment. 
Après tokenization et padding, chaque critique est représentée par une séquence de 256 indices de tokens (vocab de 10,002 mots). 
Format d'entrée : `(batch_size, 256)` de type `torch.long`. 
Sortie : 1 logit pour classification binaire avec `BCEWithLogitsLoss`.

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
| ----: | --------: | ----------------------------------------------------- |
| Train |    20 000 | (80% du train original)                               |
|   Val |     5 000 | (20% du train original)                               |
|  Test |    25 000 | (split test original)                                 |

**D2.** Donnez la taille de chaque split et le nombre de classes.

Train : 25 000 exemples (avant split) avec 20 000 pour l'entraînement après split 80/20 
Val : 5 000 exemples
Test : 25 000 exemples (split test original)
Nombre de classes : 2 classes (positif/négatif)

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

Oui, un split de validation a été créé car le dataset IMDb ne fournit que les splits train et test.

Méthode : `torch.utils.data.random_split()` sur le train original (25,000 exemples)
Ratio : 80/20 (train/val)
  Train final : 20,000 exemples
  Validation : 5,000 exemples
Seed : 42 (fixé via `torch.Generator().manual_seed(42)`)
Stratification : Non appliquée (split aléatoire simple), mais le dataset original étant parfaitement équilibré (50/50), le split résultant conserve approximativement cet équilibre.

Cette approche garantit la reproductibilité des expériences tout en préservant l'équilibre des classes.

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.

| Split      | Négatif | Positif | Total  | Balance |
| ---------- | ------: | ------: | -----: | ------: |
| Train      |   9,987 |  10,013 | 20,000 |   50.1% |
| Validation |   2,513 |   2,487 |  5,000 |   49.7% |
| Test       |  12,500 |  12,500 | 25,000 |   50.0% |

Le dataset est quasi-parfaitement équilibré (~50% par classe dans tous les splits). Cet équilibre est idéal pour l'entraînement : aucune pondération des classes n'est nécessaire, l'accuracy est une métrique fiable (baseline aléatoire à 50%), et le modèle ne sera pas biaisé vers une classe particulière. Cela permet une évaluation objective des performances sans ajustement spécifique pour gérer un déséquilibre.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

Longueurs variables : Les critiques IMDb ont des longueurs très hétérogènes (de quelques mots à plusieurs centaines). Pour gérer cela, toutes les séquences sont tronquées à 256 tokens (séquences longues coupées) ou paddées avec le token `<pad>` (séquences courtes complétées). Cette standardisation est nécessaire pour le traitement en batch par le LSTM.

Vocabulaire limité : Le vocabulaire est restreint aux 10 000 mots les plus fréquents + 2 tokens spéciaux (`<unk>`, `<pad>`). Les mots rares ou hors vocabulaire sont remplacés par `<unk>`, ce qui peut entraîner une perte d'information pour les critiques contenant un vocabulaire spécifique ou technique.

Complexité linguistique : Les critiques contiennent du sarcasme, des négations complexes ("not bad" = positif), et des expressions idiomatiques qui rendent la tâche de classification difficile, nécessitant une bonne capture du contexte par le BiLSTM bidirectionnel.

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = __, center-crop = __, normalize = (mean=__, std=__)…
- Audio : resample = __ Hz, mel-spectrogram (n_mels=__, n_fft=__, hop_length=__), AmplitudeToDB…
- NLP : tokenizer = Tokenization par regex `\b\w+\b` (extraction des mots, conversion en minuscules), vocab = 10 002, max_length = 256 tokens, padding/truncation = 256/256
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?

1. Tokenization (regex `\b\w+\b` + lowercase)*
- Opération : Conversion du texte en minuscules puis extraction des mots par expression régulière
- Pourquoi : Simplifie le vocabulaire en évitant les doublons de casse ("Movie" vs "movie") et extrait uniquement les mots alphanumériques, éliminant la ponctuation qui apporte peu d'information pour l'analyse de sentiment

2. Construction du vocabulaire (top 10,000 mots)
- Paramètres : 10,000 mots les plus fréquents + `<pad>` (index 0) + `<unk>` (index 1)
- Pourquoi : Limite la taille de la matrice d'embeddings (réduction mémoire/calcul) tout en conservant ~95% de la couverture du corpus. Les mots rares apportent peu d'information discriminante pour la classification

1. Padding/Truncation (max_length = 256)
- Opération : 
  - Séquences > 256 tokens → troncature aux 256 premiers tokens
  - Séquences < 256 tokens → ajout de `<pad>` jusqu'à 256
- Pourquoi : Nécessaire pour le traitement en batch (tenseurs de taille fixe). 256 tokens représentent un bon compromis : suffisamment long pour capturer l'essentiel d'une critique (~230 mots en moyenne) tout en restant computationnellement raisonnable pour le LSTM

4. Conversion en indices (tokens → integers)
- Opération : Mapping de chaque token vers son index dans le vocabulaire (mots inconnus → index 1)
- Pourquoi : Format d'entrée requis pour la couche `nn.Embedding` qui transforme les indices en vecteurs denses

Aucune normalisation ou augmentation n'est appliquée à ce stade (pas de synonym replacement, back-translation, etc.).

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

Non, les prétraitements sont strictement identiques pour les trois splits (train/val/test) :

- Même tokenizer : regex `\b\w+\b` + lowercase
- Même vocabulaire : construit uniquement sur le train, puis appliqué de manière identique sur val et test
- Mêmes paramètres : max_length = 256, padding/truncation identiques
- Pas d'augmentation : aucune opération aléatoire (pas de synonym replacement, dropout de mots, etc.)

Justification : En NLP comme en vision, les prétraitements déterministes (tokenization, padding) doivent être identiques pour garantir que le modèle voit les mêmes types d'entrées à l'entraînement et à l'évaluation. Le vocabulaire est figé après construction sur le train pour simuler des conditions réelles (mots inconnus en test = `<unk>`).

Seule différence : Le shuffle des DataLoaders (`shuffle=True` pour train, `shuffle=False` pour val/test), mais cela n'affecte pas les données elles-mêmes, uniquement l'ordre de présentation.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ? 

Aucune augmentation de données n'a été appliquée.

Justification :
- Pour ce projet, nous nous concentrons sur l'architecture de base du BiLSTM + Attention sans augmentation
- Les techniques d'augmentation en NLP (synonym replacement, back-translation, word dropout, etc.) sont plus complexes à implémenter et peuvent introduire du bruit sémantique
- Le dataset IMDb est suffisamment large (20,000 exemples d'entraînement) pour entraîner le modèle sans sur-apprentissage immédiat
- Le dropout (p=0.3) intégré dans le modèle joue déjà un rôle de régularisation

Augmentations possibles (non implémentées) :
- Synonym replacement : remplacement aléatoire de N mots par leurs synonymes (p=0.1, N=2-3)
- Random deletion : suppression aléatoire de mots (p=0.1)
- Random swap : permutation aléatoire de mots adjacents (p=0.1)
- Back-translation : traduction vers une langue intermédiaire puis retour en anglais

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

Aucune augmentation n'est appliquée dans ce projet, donc la question de conservation des labels ne se pose pas.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.

Retour console après lancement commande : python -m src.tests.visualize_preprocessing

--- Exemple 1 ---
Label: NÉGATIF ✗
Longueur: 256 tokens (+ 0 padding)
Texte (50 premiers mots): the fallen ones starts with <unk> matt <unk> casper van <unk> in the desert discovering the <unk> remains of a <unk> foot tall giant now there s something you don t see everyday matt is working for property <unk> <unk> robert wagner who wants to build a holiday resort on...

--- Exemple 2 ---
Label: POSITIF ✓
Longueur: 256 tokens (+ 0 padding)
Texte (50 premiers mots): this movie is not a kung fu movie this is a comedy about kung fu and if before making this film sammo hung hadn t spent some time watching films by the great french comic filmmaker <unk> <unk> i ie e g <unk> <unk> de <unk> he is certainly on...

--- Exemple 3 ---
Label: POSITIF ✓
Longueur: 182 tokens (+ 74 padding)
Texte (50 premiers mots): pleasant story of the community of <unk> in london who after an <unk> ww2 bomb explodes find a royal <unk> stating that the area they live in forms part of <unk> br br this movie works because it appeals to the fantasy a lot of us have about making up...

Observation : Les critiques conservent clairement leur sentiment après preprocessing (vocabulaire positif/négatif intact), les longueurs variables sont correctement normalisées à 256 tokens via padding/truncation, et les quelques tokens `<unk>` (mots rares) n'altèrent pas le sens global. Le preprocessing transforme efficacement le texte brut en séquences numériques exploitables par le BiLSTM tout en préservant l'information sémantique nécessaire à la classification.

**D11.** Donnez la **forme exacte** d'un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

Forme d'un batch d'entrée (inputs) :
- Shape : `(batch_size, 256)` ou plus généralement `(batch_size, seq_len)`
- Exemple concret avec batch_size=64 : `torch.Size([64, 256])`
- Dtype : `torch.long` (indices de tokens entiers)
- Range : `[0, 10001]` (indices du vocabulaire)

Forme d'un batch de labels :
- Shape : `(batch_size,)` 
- Exemple concret avec batch_size=64 : `torch.Size([64])`
- Dtype : `torch.float32` (pour BCEWithLogitsLoss)
- Range : `{0.0, 1.0}` (0=négatif, 1=positif)

Vérification de cohérence avec `meta["input_shape"]` :
- `meta["input_shape"]` = `(256,)`
- Batch shape = `(batch_size, 256)`
- Cohérence vérifiée : la dimension de séquence (256) correspond bien à `meta["input_shape"][0]`

Résumé :
```python
# Un batch typique
inputs.shape  # torch.Size([64, 256])  - 64 critiques de 256 tokens
labels.shape  # torch.Size([64])        - 64 labels binaires
meta["input_shape"]  # (256,)          - longueur de séquence attendue
```

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `Accuracy, F1-macro` → score = `0.4974, 03322`
- **Prédiction aléatoire uniforme** — Métrique : `Accuracy, F1-macro` → score = `0.5144, 0.5144`  
_Commentez en 2 lignes ce que ces chiffres impliquent._

La baseline "classe majoritaire" donne une accuracy proche de 50% mais un F1 macro faible (~33%), indiquant un biais vers une seule classe. En revanche, la baseline "prédiction aléatoire uniforme" atteint un F1 macro de ~50%, ce qui représente un plancher minimal pour un dataset équilibré.

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → (batch_size, 256)
  - Couche initiale : Embedding(vocab_size=10,002, embedding_dim=300, padding_idx=1)
  - Stage 1 (répéter N₁ fois) : LSTM(input_size=300, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
  - Stage 2 (répéter N₂ fois) : AttentionLayer(hidden_size=256)
  - Tête (GAP / linéaire) → logits (dimension = nb classes) : Dropout(p=0.3) / Linear(in_features=256, out_features=1)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = (batch_size, 1)

- **Nombre total de paramètres** : `3 836 698`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).

1. hidden_size=128 :
  - Rôle : Définit la taille des états cachés du LSTM, influençant la capacité du modèle à capturer des dépendances complexes dans les séquences.
  - Impact : Une valeur plus grande permet de capturer des motifs plus riches, mais augmente le risque de surapprentissage et les besoins en mémoire.

2. num_layers=2 :
  - Rôle : Nombre de couches LSTM empilées, permettant d'apprendre des représentations plus abstraites à chaque couche.
  - Impact : Plus de couches augmentent la profondeur du modèle, mais nécessitent plus de données pour éviter le surapprentissage.

Ces choix permettent un bon compromis entre expressivité et stabilité, tout en respectant les contraintes du projet.

### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `0.6820`
- **Vérification** : backward OK, gradients ≠ 0 (norme totale = `0.5527`)

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

La loss initiale observée est cohérente avec la valeur attendue pour une classification binaire utilisant `BCEWithLogitsLoss`. Les logits initiaux étant proches de 0, la perte est proche de $-\log(1/2) \approx 0.693$. La rétropropagation a été vérifiée, et les gradients ne sont pas nuls.

- **Forme du batch** : `(64, 256)` (inputs)
- **Forme de sortie du modèle** : `(64, 1)` (logits)

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR  | WD  | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
| ------------------- | --- | --- | ----- | ----- | ---------------------- | -------- | ----- |
|                     |     |     |       |       |                        |          |       |
|                     |     |     |       |       |                        |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


