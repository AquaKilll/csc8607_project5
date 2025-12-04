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
- NLP : tokenizer = Tokenization par regex `\b\w+\b` (extraction des mots, conversion en minuscules), vocab = 10,002 tokens (10,000 mots les plus fréquents + `<unk>` + `<pad>`)
  - `<pad>` : index 0 (pour padding)
  - `<unk>` : index 1 (pour mots inconnus), max_length = 256 tokens, padding/truncation = Séquences > 256 tokens : **troncature** à 256 / Séquences < 256 tokens : **padding** avec token `<pad>` (index 0)
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  
**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `_____` → score = `_____`
- **Prédiction aléatoire uniforme** — Métrique : `_____` → score = `_____`  
_Commentez en 2 lignes ce que ces chiffres impliquent._

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

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


