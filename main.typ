// Documentation technique de référence
// Architecture complète du projet de quantification adaptative et de profilage matériel

#import "report-theme.typ": report-accent, report-accent2, report-accent3, report-theme
#import "helpers.typ": callout, warning-box, tip-box, status

#show: report-theme.with(
  title: "Architecture complète du projet de quantification adaptative et de profilage matériel",
  author: "Équipe Projet",
  rhythm: "report",
  running-header: true,
)

// ============================================================================
// PAGE DE TITRE
// ============================================================================
#page(margin: (top: 25%, x: 2.2cm, bottom: 2.2cm), numbering: none, header: none, footer: none)[
  #set par(first-line-indent: 0em)
  #align(center)[
    #box(
      fill: rgb("#eef4fb"),
      inset: (x: 10pt, y: 5pt),
      radius: 3pt,
      text(size: 9pt, weight: "bold", fill: report-accent)[DOCUMENTATION TECHNIQUE DE RÉFÉRENCE],
    )
    #v(1em)
    #text(size: 21pt, weight: "bold", fill: report-accent, hyphenate: false)[Architecture complète du projet de quantification adaptative et de profilage matériel]
    #v(0.6em)
    #text(size: 13pt, fill: luma(90))[Profiler V3/V5 — Quantification Algo — Z2/D2 Planner — Application Kimi K3]
    #v(2.2em)
    #line(length: 35%, stroke: 0.8pt + rgb("#007f8c"))
    #v(2.2em)
    #text(size: 11pt, fill: luma(80))[
      #grid(
        columns: (auto, auto),
        column-gutter: 20pt,
        row-gutter: 6pt,
        align: (right, left),
        text(weight: "bold")[Auteur(s) :], [Équipe Projet],
        text(weight: "bold")[Date :], [7 novembre 2026],
        text(weight: "bold")[Version :], [1.0 — Architecture de référence],
        text(weight: "bold")[Statut :], [Document de conception vivant],
      )
    ]
  ]
  #align(bottom + center)[
    #text(size: 8.5pt, fill: luma(140))[Quantification adaptative · Profilage matériel · Modèles MoE · Working-set contraint]
  ]
]

// ============================================================================
// TABLE DES MATIÈRES
// ============================================================================
#page(numbering: none, header: none, footer: none)[
  #show outline.entry: set text(size: 8.8pt)
  #outline(title: [Table des matières], indent: 1.3em)
]

#counter(page).update(1)

// ============================================================================
// 1. VISION GÉNÉRALE DU PROJET
// ============================================================================
= Vision générale du projet

L'objectif est de développer un *système complet* capable d'analyser, quantifier, profiler, planifier et exécuter des modèles de langage dont la taille dépasse largement la mémoire physique disponible.

Le principe fondamental est de *ne pas chercher uniquement à réduire le nombre de bits par poids*. Une approche qui se limiterait à compresser aveuglément les poids ignore les contraintes réelles d'exécution.

Il faut optimiser simultanément l'ensemble des dimensions suivantes :

- La précision numérique.
- La taille physique du modèle.
- La qualité de sortie.
- La mémoire résidente.
- Les transferts de données.
- La bande passante mémoire et stockage.
- La capacité de calcul des accélérateurs.
- La répartition CPU / GPU / NPU.
- Le coût de déquantification.
- Le comportement des modèles denses et MoE (Mixture of Experts).
- La latence et le débit réel en tokens par seconde.

#tip-box("Exigence clé")[
Le système doit être capable de prendre un modèle significativement plus grand que la mémoire disponible et de déterminer automatiquement *comment l'exécuter avec un working-set strictement limité*.
]

== Les trois composants principaux

#table(
  columns: (1.4fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (left, left),
  [Composant], [Responsabilité principale],
  [Quantification Algo], [Recherche de représentations numériques optimales pour chaque tenseur],
  [Profiler V3/V5], [Analyse, mesure, prédiction et validation sur le matériel cible],
  [Z2/D2 Planner], [Décision, allocation des ressources et stratégie d'exécution],
)

Le *runtime* constitue le quatrième élément de validation : il permet de confronter les prédictions du profiler aux performances réellement obtenues et d'alimenter une boucle de feedback d'adaptation.

== Philosophie

Une quantification n'est pas bonne simplement parce qu'elle produit un fichier plus petit. Elle doit être évaluée selon son *coût total* :

$ C = f(Q, M, upright(I O), D, K, L) $

avec :
- $Q$ : qualité numérique préservée ;
- $M$ : empreinte mémoire ;
- $upright(I O)$ : coût des transferts (stockage $arrow.r$ mémoire) ;
- $D$ : coût de déquantification ;
- $K$ : coût de calcul GEMM / kernels ;
- $L$ : latence perçue par l'utilisateur.

#callout("Théorème informel du projet")[
Le meilleur format dépend du matériel, du tenseur, de la couche et du contexte d'exécution. Il n'existe pas de format universel optimal.
]

#figure(
  image("figures/architecture.png", width: 95%),
  caption: [Vue d'ensemble de l'architecture : les quatre composants et leurs flux d'information.],
)

// ============================================================================
// 2. PROFILER — AXES DE PROFILAGE
// ============================================================================
= Profiler — Axes de profilage

Le *Profiler V3/V5* est le système de mesure et de prédiction du projet. Il articule douze axes de profilage complémentaires.

== Axe 1 — Cartographie structurelle du modèle

*Objectif :* connaître exactement la structure du modèle avant toute optimisation. Le profiler doit reconstruire l'organisation réelle des poids et des opérations.

*Informations collectées :*
- Nombre de couches ;
- Dimensions des tenseurs ;
- Nombre de paramètres ;
- Types de tenseurs ;
- Dimensions d'entrée et de sortie ;
- Architecture attentionnelle (MHA, GQA, MLA) ;
- Embeddings et LM Head ;
- MLP ;
- Normalisations (RMSNorm, LayerNorm) ;
- Attention dense ou latente ;
- Couches récurrentes ;
- Nombre d'experts et Top-$K$ du routeur ;
- Organisation des blocs MoE.

*Granularité :* le système doit analyser quatre niveaux — (1) modèle complet, (2) couche, (3) tenseur, (4) famille de tenseurs. La *granularité expert* constitue une extension nécessaire pour les modèles MoE.

*Sources :* GGUF, Safetensors, checkpoints Hugging Face, métadonnées d'architecture, structures de tenseurs et formats propriétaires documentés.

*Résultats produits :* `model_map.json`, `layer_map.csv`, `tensor_map.csv`, `architecture_report.json`.

*État :* #status("implemented") analyse GGUF, couches, tenseurs, familles, dimensions et informations MoE structurelles. *À approfondir :* cartographie physique individuelle des experts et association avec leur identifiant réel.

== Axe 2 — Analyse des formats numériques

*Objectif :* identifier ce que chaque format coûte réellement. Le profiler ne doit pas se limiter à lire l'étiquette Q4 ou Q8 : il doit comprendre la représentation physique.

*Formats à prendre en charge :* BF16, FP16, FP32, INT8, Q8_0, Q6, Q5, Q4, Q3, Q2, Q1_0, formats IQ, MXFP4, E8M0, ainsi que les formats expérimentaux propres au projet.

*Analyse par format :* pour chaque représentation sont quantifiés les bits par poids effectifs, la taille du bloc, le nombre de valeurs par bloc, la taille des échelles, les métadonnées, l'alignement mémoire, le padding, les dépendances du décodeur, la compatibilité CPU / GPU / NPU et le coût de déquantification.

#tip-box("Exemple : format Q1_0 (Bonsai-27B)")[
Le format Q1_0 étudié utilise des blocs de 128 éléments codés sur 18 octets, soit *1,125 bit par poids* effectif. Le profiler doit distinguer le coût théorique (1,125 bpw) du coût réel (overhead des échelles, temps de conversion).
]

*Résultats :* `quant_formats.py`, `format_capability_report.json`.

*État :* #status("implemented") registre de formats et analyse de plusieurs formats GGUF et formats de recherche. *Extension :* intégrer les formats propres aux modèles MoE et les formats compressés sans perte.

== Axe 3 — Sensibilité à la quantification

*Objectif :* découvrir quels poids peuvent être fortement quantifiés et lesquels doivent conserver une précision supérieure. Une quantification uniforme suppose que tous les poids ont une importance équivalente — hypothèse généralement fausse.

*Mesures par couche et par tenseur :*
- Distribution des poids (moyenne, écart-type, valeurs extrêmes) ;
- Erreur de reconstruction : RMS error, erreur maximale, *cosine similarity*, SNR ;
- Sensibilité aux changements de précision.

*Méthodes :* reconstruction par quantification, comparaison avec les poids originaux, analyse des distributions, calibration sur activations, comparaison de sorties et analyse de sensibilité par couche.

*Fonction objectif de sensibilité :*

$ E_i = frac(||W_i - hat(W)_i||_2, ||W_i||_2 + epsilon) $

Cette erreur n'est qu'un indicateur : elle ne remplace pas la mesure de l'impact sur les activations et les logits.

*Résultats :* `layer_sensitivity.csv`, `tensor_sensitivity.csv`.

*État :* #status("implemented") analyses de reconstruction et mesures RMS, maximum et cosine. *Extension :* calibration systématique par activation et sensibilité aux tâches.

== Axe 4 — Allocation adaptative de précision

*Objectif :* sélectionner automatiquement la précision de chaque partie du modèle. C'est le cœur algorithmique de Quantification Algo.

Au lieu d'imposer un Q4 global, le système peut sélectionner :
- Q2 pour certains tenseurs peu sensibles ;
- Q3 pour d'autres ;
- Q4 pour les couches intermédiaires ;
- Q6 ou Q8 pour les couches sensibles ;
- BF16 pour les opérations qui doivent conserver leur précision intacte.

*Variables de décision par tenseur $i$ :* taille originale $S_i$, taille quantifiée $S_(i,q)$, erreur $E_(i,q)$, coût de calcul $C_(i,q)$ et coût de transfert $T_(i,q)$. Le système doit respecter un budget mémoire :

$ sum_i S_(i, q_i) <= B $

tout en minimisant la dégradation globale :

$ min sum_i w_i E_(i, q_i) $

où $w_i$ représente l'importance relative de la partie du modèle. Une version avancée intègre également le temps de calcul et de transfert dans l'objectif.

*Algorithmes envisageables :* allocation gloutonne, programmation dynamique, sac à dos multidimensionnel, optimisation sous contraintes, recherche locale, allocation basée sur Pareto, calibration adaptative.

*Sortie :* `quantization_plan.json` conservant chaque décision avec sa justification.

*État :* #status("implemented") planification de quantification par couche, tenseur et famille. *Extension :* optimisation multiobjectif intégrant simultanément qualité, résidence, bande passante et calcul.

== Axe 5 — Profilage mémoire

*Objectif :* comprendre la différence entre taille du modèle et mémoire réellement nécessaire pour l'exécuter. Il faut séparer trois budgets distincts :

#table(
  columns: (1fr, 3.5fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  [Budget], [Définition],
  [*A. Storage budget*], [Taille totale du modèle sur le stockage (SSD / distant)],
  [*B. Memory budget*], [Mémoire disponible pour les poids, activations, caches et autres allocations],
  [*C. Movement budget*], [Quantité de données qu'il faut déplacer pendant l'exécution],
)

*Mesures :* RAM totale et disponible, VRAM, working set, poids résidents, chargés et demandés, activations, KV cache, cache des experts, pic mémoire, fragmentation, page cache, mémoire DMA et VTCM.

*Objectif concret :* être capable d'exécuter un modèle supérieur à 8 GiB avec un working set inférieur ou égal à 8 GiB.

*État :* #status("implemented") estimations mémoire, résidence et plafonds de stockage dans V5. *Extension :* instrumentation plus précise de la mémoire physique, des allocations temporaires et du comportement page cache.

== Axe 6 — Profilage I/O et transferts

*Objectif :* mesurer le prix réel de la lecture des poids — déterminant pour les modèles qui dépassent la RAM disponible.

*Mesures :* octets lus à froid / à chaud par token, nombre de requêtes, latence des lectures, débit SSD, cache hit/miss, lectures répétées, coût de décompression, temps de transfert, coût du mapping, DMA et RPC vers accélérateur.

*Méthodes :* lecture sélective de tenseurs, HTTP Range, streaming de poids, analyse des pages et comparaison des accès froids et chauds.

*Métrique centrale :*

$ upright(I O)_(t o k e n) = frac(upright("Octets effectivement lus"), upright("Tokens générés")) $

#warning-box("Règle impérative I/O")[
Il faut impérativement distinguer les octets demandés, les octets réellement lus depuis le stockage et les octets déjà disponibles dans le page cache.
]

*État :* #status("implemented") estimation *cold bytes/token*, résidence, plafonds et lecture HTTP Range de tenseurs. *Extension :* mesure live complète du stockage et attribution des transferts par couche et par expert.

== Axe 7 — Profilage CPU / GPU / NPU

*Objectif :* déterminer où chaque opération doit réellement s'exécuter. Une opération théoriquement compatible avec un accélérateur n'est pas nécessairement exécutée sur celui-ci.

*CPU :* SIMD, AVX2, FMA, NEON, déquantification, GEMM, opérations scalaires.

*GPU :* CUDA, OpenCL, occupation, bande passante, temps kernel, transferts host/device, détection des fallbacks CPU.

*Qualcomm HTP :* QAIRT, HTP v81, VTCM, DMA, RPC, opérations supportées, placement réel des couches et fallbacks.

*Adreno :* OpenCL, kernels MUL_MAT, copies, temps d'exécution, compatibilité des formats.

*Mesure indispensable :*

$ T_(t o t a l) = T_(l o a d) + T_(t r a n s f e r) + T_(d e q u a n t) + T_(c o m p u t e) + T_(s y n c) $

#warning-box("Principe fondamental de placement")[
Ne jamais assimiler la compatibilité d'une opération à son accélération effective.
]

*État :* #status("implemented") base de capacités, analyse des formats, télémétrie Android, parsing HTP et OpenCL, analyses de copies et de placements. *Extension :* attribution systématique des coûts par opération et par tenseur.

== Axe 8 — Profilage spécifique MoE

*Objectif :* passer d'une analyse de conteneurs d'experts à une véritable analyse de leur *comportement*. C'est l'axe à développer en priorité pour Kimi K3.

*Informations structurelles :* nombre total d'experts, dimensions, experts activés, Top-$K$, taille physique par expert, format numérique, organisation des poids et identifiants.

*Informations dynamiques :* expert sélectionné, fréquence de sélection, distribution de routage, réutilisation, cache hit/miss, évictions, temps de chargement, octets transférés, temps de calcul et impact sur la latence.

*Indicateurs :*

$ upright(H i t R a t e) = frac(upright(C a c h e H i t s), upright(C a c h e H i t s) + upright(C a c h e M i s s e s)) $

$ upright(E x p e r t R e u s e)(e) = frac(N_e, N_(t o t a l)) $

Le profiler doit également mesurer l'entropie de distribution des experts : une distribution uniforme et une distribution concentrée n'impliquent pas du tout la même stratégie de cache.

*État :* #status("partial") informations structurelles MoE, nombre d'experts et Top-$K$ présents. *Non implémenté de manière complète :* identifiants dynamiques, routage réel, cache par expert, évictions, résidence et quantification individuelle des experts.

== Axe 9 — Profilage KV cache et contexte

*Objectif :* mesurer le coût de la mémoire de contexte — le poids du modèle n'est pas le seul facteur de consommation.

*Mesures :* tokens de contexte, taille KV par token, nombre de couches, de têtes, dimension des têtes, précision du KV (FP16, BF16, INT8, INT4), croissance mémoire, cache MLA, états récurrents, temps de lecture/écriture.

*Indicateur :*

$ M_(K V)(L) approx L times M_(K V \/ t o k e n) $

Pour les architectures MLA ou récurrentes, le calcul doit être adapté à la structure réelle du modèle.

*État :* #status("implemented") déjà étudié dans les scénarios de capacité mémoire et de contextes longs. *Extension :* instrumentation dynamique homogène sur les différents backends.

== Axe 10 — Profilage de performance et latence

*Objectif :* expliquer chaque seconde consommée pendant la génération.

*Mesures :* prefill tok/s, decode tok/s, temps par token, latence initiale, moyenne, P50, P95, P99, variance, temps de chargement, de déquantification, GEMM, transferts et synchronisations.

#callout("Règle de publication des débits")[
Ne jamais publier un seul débit sans préciser : le matériel, le backend, le format, le contexte, le nombre de répétitions, l'état du cache (froid ou chaud) et la méthode de mesure.
]

*État :* #status("implemented") mesures expérimentales et outils de profilage disponibles. *Extension :* corrélation temporelle complète par opération et par token.

== Axe 11 — Profilage de qualité

*Objectif :* garantir que les économies de mémoire ne détruisent pas les capacités du modèle.

*Métriques :* perplexité, cross-entropy, KL divergence, RMS error, erreur maximale, *cosine similarity*, SNR, *logit agreement*, *token agreement*, qualité de génération et stabilité du routage.

*Niveaux de validation :*
- *L1 — Structure et reconstruction :* poids, dimensions, formats, déquantification, erreurs numériques.
- *L2 — Comportement des opérations :* sorties intermédiaires, activations, logits, opérations critiques.
- *L3 — Validation du modèle :* perplexité, générations, comportement global et performances.

#warning-box("Règle de validation")[
La réussite de L1 ne démontre pas la réussite de L3.
]

*État :* #status("implemented") mesures de reconstruction et de qualité, architecture L1–L3. *Extension :* protocole uniforme sur tous les formats et backends.

== Axe 12 — Prédiction des performances

*Objectif :* prédire les performances avant de lancer une quantification ou une exécution coûteuse. Le profiler doit devenir un *modèle prédictif matériel*.

*Entrées :* architecture, formats, dimensions, quantification, RAM disponible, VRAM, bande passante, débit stockage, capacité CPU/GPU/NPU, cache, longueur du contexte.

*Sorties :* taille finale, working set, pic mémoire, octets lus par token, temps de déquantification, temps de calcul, débit estimé, latence, risque de saturation.

*Modèle simplifié :*

$ T_(t o k e n) = T_(upright(I O)) + T_(c o m p u t e) + T_(d e q u a n t) + T_(s y n c) + T_(o t h e r) $

Les temps ne sont pas toujours simplement additifs : le modèle final doit tenir compte du *chevauchement* entre transferts et calculs.

*État :* #status("implemented") prédictions de capacité, de mémoire et de coûts. *Extension :* modèle calibré par backend, capable de prédire avec des intervalles d'incertitude.

#figure(
  image("figures/memory_model.png", width: 72%),
  caption: [Modèle de mémoire à trois budgets et flux de streaming des poids et experts.],
)

// ============================================================================
// 3. QUANTIFICATION ALGO — AXES ALGORITHMIQUES
// ============================================================================
= Quantification Algo — Axes algorithmiques

== Algorithme 1 — Allocation de précision mixte

Le système choisit une précision *différente pour chaque couche et chaque tenseur* au lieu d'imposer un Q4 uniforme.

*Fonctionnement en 10 étapes :* (1) charger les poids source ; (2) identifier les tenseurs ; (3) calculer leur sensibilité ; (4) générer plusieurs candidats de quantification ; (5) mesurer leur erreur ; (6) calculer leur taille ; (7) estimer leur coût matériel ; (8) construire un plan ; (9) comparer les solutions Pareto ; (10) produire le checkpoint final.

*Objectif :* maximiser la qualité sous un budget mémoire et un budget de latence donnés.

== Algorithme 2 — Quantification sensible aux activations

La quantification des poids doit être complétée par l'analyse des activations : deux tenseurs ayant une erreur de reconstruction similaire peuvent produire des effets très différents sur les sorties.

*Méthodes candidates :* calibration sur activations, GPTQ, AWQ, SmoothQuant, recherche de clipping, optimisation des échelles, *group-wise quantization*. Le système collecte des activations représentatives, teste les candidats et mesure l'effet sur les sorties, en respectant l'architecture et les contraintes de chaque format.

== Algorithme 3 — Optimisation des échelles et des groupes

L'objectif est de choisir conjointement : la taille des groupes, les échelles, le zéro-point (lorsque le format le permet), le clipping, la distribution des valeurs et l'alignement des blocs. Le choix doit être évalué selon la précision et le coût d'exécution réel — une quantification à 4 bits avec des métadonnées trop lourdes peut être moins efficace qu'attendu.

== Algorithme 4 — Quantification hiérarchique

Le système doit pouvoir optimiser successivement :
(1) famille de tenseurs $arrow.r$ (2) couche $arrow.r$ (3) tenseur $arrow.r$ (4) bloc $arrow.r$ (5) expert $arrow.r$ (6) éventuellement sous-structure d'expert.

Cette hiérarchie limite le coût de recherche : seules les parties sensibles sont affinées avec une granularité supérieure.

== Algorithme 5 — Quantification MoE par expert

Extension particulièrement importante pour Kimi K3 : chaque expert peut recevoir sa propre allocation de précision.

*Variables prises en compte :* sensibilité de l'expert, fréquence d'activation, importance dans le routage, taille, coût de chargement et de calcul, budget mémoire.

*Stratégie :* un expert fréquemment utilisé et sensible conserve une précision élevée ; un expert peu utilisé et robuste reçoit une précision plus basse. *Attention :* la fréquence d'utilisation ne suffit pas, il faut également mesurer la sensibilité et l'impact sur les logits.

*Résultat :* `expert_quantization_plan.json`.

*État :* #status("development") axe de développement, non validé dans V5.

== Algorithme 6 — Compression physique sans perte

Il faut distinguer deux opérations fondamentalement différentes :

- *Modifier les valeurs numériques* par quantification (avec perte contrôlée) ;
- *Compresser la représentation physique* sans modifier les valeurs (sans perte).

Pour les experts MXFP4, cette distinction est essentielle. L'objectif est de réduire les lectures SSD et les transferts sans introduire d'erreur numérique supplémentaire, en décompressant au chargement ou pendant le streaming.

*Mesures :* ratio de compression, débit de décompression, CPU consommé, octets physiques lus vs octets logiques reconstruits, impact sur la latence.

== Algorithme 7 — Optimisation du cache MoE

Le cache LRU classique n'est pas nécessairement optimal pour les experts MoE.

*Stratégies à comparer :* LRU, LFU, cache adaptatif, prédiction du prochain expert, préchargement, cache pondéré par taille, cache tenant compte du coût de chargement.

*Objectif :* minimiser $T_(c a c h e) = T_(m i s s) + T_(l o a d) + T_(e v i c t i o n)$ tout en respectant le budget mémoire. La décision doit s'appuyer sur les *traces de routage réelles* et non sur une hypothèse de réutilisation.

== Algorithme 8 — Placement matériel adaptatif

Le système doit sélectionner le backend approprié pour chaque opération : CPU vs GPU, CPU vs HTP, OpenCL vs CUDA, format directement supporté vs conversion, exécution locale vs transfert.

*Critère clé :* ne pas choisir le backend uniquement selon son débit théorique. Le coût complet inclut les transferts, la conversion, les synchronisations et les éventuels fallbacks — une opération plus rapide sur GPU peut être plus lente une fois le transfert pris en compte.

== Algorithme 9 — Optimisation du working set

Le système construit un plan d'exécution compatible avec une limite mémoire imposée : quels poids restent résidents, quels poids sont chargés à la demande, quels experts sont préchargés, quelles couches peuvent être streamées, quelle quantité de KV cache peut être conservée et quelle réserve mémoire maintenir. Le but n'est pas seulement de faire tenir les poids, mais d'éviter que les transferts détruisent le débit.

== Algorithme 10 — Recherche multiobjectif et Pareto

Le système doit éviter de rechercher une solution unique avant de connaître les compromis possibles. Les axes d'optimisation sont : qualité, taille, latence, débit, RAM, VRAM, I/O, énergie et compatibilité matérielle.

Une *solution Pareto* est une solution pour laquelle aucun objectif ne peut être amélioré sans dégrader au moins un autre objectif. Le planner sélectionne ensuite la solution la plus adaptée aux contraintes réelles de l'appareil cible.

// ============================================================================
// 4. Z2/D2 PLANNER — PLANIFICATION ET DÉCISION
// ============================================================================
= Z2/D2 Planner — Planification et décision

Le planner exploite les résultats du profiler et de Quantification Algo. Il ne remplace pas le profiler : *il consomme ses observations et ses prédictions* pour construire des décisions reproductibles et justifiées.

#table(
  columns: (0.4fr, 1.2fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  [Étape], [Nom], [Description],
  [1], [Capability discovery], [Identifier les ressources matérielles disponibles (CPU, GPU, NPU, RAM, débits)],
  [2], [Cost model], [Calculer les coûts mémoire, I/O, calcul et transfert],
  [3], [Candidate generation], [Générer plusieurs plans de quantification et d'exécution alternatifs],
  [4], [Static Oracle], [Comparer les plans selon les informations connues avant exécution],
  [5], [Prediction], [Estimer le débit, la latence et la qualité de chaque plan],
  [6], [Allocation], [Répartir les poids, les caches et les opérations sur les ressources],
  [7], [Execution planning], [Construire la stratégie d'exécution ordonnancée],
  [8], [Feedback], [Comparer les performances prédites aux mesures réelles en cours d'exécution],
  [9], [Adaptation], [Réviser les décisions lorsque les observations contredisent les prédictions],
)

#figure(
  image("figures/planner_pipeline.png", width: 58%),
  caption: [Le pipeline de décision en 9 étapes du Z2/D2 Planner avec boucle d'adaptation.],
)

Le planner doit *conserver les décisions et leurs justifications* afin de garantir la reproductibilité et la traçabilité de chaque choix d'optimisation.

// ============================================================================
// 5. VALIDATIONS EXPÉRIMENTALES DÉJÀ RÉALISÉES
// ============================================================================
= Validations expérimentales déjà réalisées

== GTX 1080 — Qwen3.5-9B

Machine de test : NVIDIA GeForce GTX 1080 8 Go. Résultats documentés :

#table(
  columns: (2fr, 1.3fr, 1.3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (left, center, center),
  [Mesure], [Format mixte], [Q4 uniforme],
  [Taille GPU], [5,388 GiB], [5,079 GiB],
  [Perplexité], [9,7839], [10,0437],
  [Débit], [33,42 tok/s], [35,98 tok/s],
)

*Enseignement majeur :* le format mixte améliore significativement la perplexité (+2,6 % de qualité) au prix d'une baisse très modérée de débit. Travaux associés : 442 tenseurs analysés, `tensor_map.csv`, analyse PPL, SNR, coûts par couche et par tenseur, variantes v7/v8/v4/v9/Q8_0. *Cette expérience constitue une validation concrète de l'intérêt de la précision non uniforme.*

== GTX 1080 Ti — Bonsai-27B

Configuration ultra-compacte Q1_0 : 1,125 bit par poids, blocs de 128 éléments, 18 octets par bloc.

#table(
  columns: (2fr, 2fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (left, left),
  [Poids seuls], [~3,9 Go],
  [Modèle + contexte 4K], [~5,2 Go],
  [Modèle + contexte 10K], [~5,6 Go],
  [Modèle + contexte 100K (KV 4 bits)], [~6,8 Go],
  [Débit rapporté], [~28,3 tok/s],
)

Ces résultats démontrent l'intérêt d'étudier les représentations extrêmement compactes. *Ils ne doivent toutefois pas être confondus avec les performances attendues sur Kimi K3.*

== MTP et décodage spéculatif

Résultats rapportés sur les expérimentations GTX 1080 :
- Environ 55–56 tok/s avec MTP ;
- Environ 61–61,3 tok/s avec une profondeur de 3 ;
- Taux d'acceptation autour de 0,702.

Cet axe doit rester distinct de la quantification : il agit sur le nombre de tokens acceptés par étape de décodage et *peut amplifier les gains* obtenus par une meilleure représentation des poids.

== Qualcomm HTP et Adreno

Travaux réalisés sur les capacités Qualcomm : analyse QAIRT, HTP v81, analyse VTCM, parsing des placements, analyse des copies, OpenCL Adreno, analyse des opérations `MUL_MAT`, détection des fallbacks CPU.

#table(
  columns: (2.5fr, 1.5fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (left, center),
  [Modèle / Backend], [Débit mesuré],
  [Marco MoE — Adreno OpenCL], [31,23 tok/s],
  [Marco MoE — HTP], [23,6 tok/s],
  [Modèle 9B — Adreno OpenCL], [10,19 tok/s],
  [Modèle 9B — HTP], [~9,8 tok/s],
  [Modèle 1,7B — Adreno OpenCL], [38,3 – 38,95 tok/s],
  [Modèle 1,7B — HTP], [32,5 tok/s],
)

#warning-box("Cas d'école : Bonsai-27B sur accélérateur mobile")[
Une expérimentation Bonsai-27B a montré un *fallback CPU de 99,7 %* sur les opérations MUL_MAT, avec seulement *0,32 tok/s mesuré contre 13,9 tok/s estimés initialement*. C'est un exemple essentiel de l'écart entre compatibilité théorique et accélération réelle.
]

// ============================================================================
// 6. APPLICATION AU KIMI K3
// ============================================================================
= Application au Kimi K3

Kimi K3 constitue un *nouveau cas d'utilisation* pour l'ensemble du système. Les dimensions clés à étudier sont : tronc dense BF16, experts MXFP4, routage Top-16, cache des experts, lecture à la demande, quantification mixte, décompression physique, prédiction du débit et allocation mémoire.

== Priorités de travail

#table(
  columns: (1fr, 1.6fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (center, left, left),
  [Priorité], [Cible], [Actions à mener],
  [*P1*], [Tronc dense BF16], [Tester : BF16 de référence, INT8, INT4, mixte INT4/INT8, allocation adaptative par couche et par tenseur],
  [*P2*], [Experts MXFP4], [Tester : conservation MXFP4, compression sans perte, quantification expérimentale supplémentaire, cache adaptatif, sensibilité par expert],
  [*P3*], [Simulation working set], [Construire des simulations adaptées aux trois plateformes cibles matérielles],
)

== Plateformes matérielles cibles pour la simulation

1. *Station intermédiaire :* Ryzen 5 5600 · 32 Go RAM · NVIDIA GeForce RTX 3060 12 Go.
2. *Station moderne :* Ryzen AI 9 HX365 · 32 Go RAM · NVIDIA GeForce RTX 5070 8 Go.
3. *Mobile haut de gamme :* OnePlus 15 · Qualcomm HTP v81 · GPU Adreno.

#warning-box("Condition d'utilisabilité des accélérateurs")[
Les accélérateurs ne doivent être considérés comme utilisables que si leurs kernels et leurs chemins de transfert sont *réellement validés* sur le matériel concerné.
]

// ============================================================================
// 7. ARCHITECTURE LOGICIELLE CIBLE
// ============================================================================
= Architecture logicielle cible

Le système est découpé en quatorze modules modulaires. Les nouveaux noms sont des propositions d'organisation cible.

#table(
  columns: (1.6fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  [Module], [Fonction principale],
  [`capability_db.py`], [Base de données des capacités matérielles (CPU, GPU, NPU, mémoires, débits)],
  [`quant_formats.py`], [Registre et propriétés physiques des formats numériques],
  [`profile_model.py`], [Cartographie structurelle des modèles (GGUF, Safetensors, HF)],
  [`profiler.py`], [Mesures mémoire, I/O et analyse de la résidence],
  [`predict_from_hf.py`], [Prédiction de performances depuis les modèles Hugging Face],
  [`predict_from_hf_v5.py`], [Prédiction avancée V5 avec plafonds de stockage et budgets],
  [`parse_hexagon.py`], [Analyse des traces d'exécution Qualcomm HTP (VTCM, fallbacks)],
  [`quantization_engine.py`], [Génération des candidats de quantification et reconstruction],
  [`sensitivity_engine.py`], [Calcul de la sensibilité numérique par couche et par tenseur],
  [`mixed_precision.py`], [Allocation optimale de précision sous contraintes mémoire],
  [`expert_profiler.py`], [Profilage dynamique MoE (experts, routage, fréquences)],
  [`expert_cache.py`], [Simulation et optimisation du cache d'experts (LRU, LFU, adaptatif)],
  [`cost_model.py`], [Modèle de coûts d'exécution (IO + compute + dequant)],
  [`planner.py`], [Pipeline de décision Z2/D2 (découverte $arrow.r$ allocation)],
  [`validation.py`], [Vérification multi-niveaux L1–L3 et métriques de qualité],
)

// ============================================================================
// 8. MÉTRIQUES DE RÉUSSITE DU PROJET
// ============================================================================
= Métriques de réussite du projet

Le projet doit produire un rapport complet pour chaque expérience menée.

#table(
  columns: (1fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  [Domaine], [Métriques à reporter],
  [*Qualité*], [Perplexité, KL divergence, logit agreement, token agreement, erreur de reconstruction RMS/max],
  [*Mémoire*], [Taille totale, working set, RSS, VRAM, KV cache, résidence des experts],
  [*I/O*], [Octets/token (froids/chauds), cache hit rate, débit stockage, temps de chargement, décompression],
  [*Calcul*], [Temps kernel, temps GEMM, temps de déquantification, % de fallback CPU, occupation GPU/HTP],
  [*Performance*], [Prefill tok/s, decode tok/s, latence P50/P95, variance, tok/s/W si mesure énergétique],
  [*Prédiction*], [Écart prédiction/réalité, intervalle de confiance, erreur par backend et par format],
)

// ============================================================================
// 9. ORDRE DE DÉVELOPPEMENT RECOMMANDÉ
// ============================================================================
= Ordre de développement recommandé

#table(
  columns: (0.7fr, 1.8fr, 3fr),
  inset: 7pt,
  stroke: 0.4pt + luma(200),
  align: (center, left, left),
  [Priorité], [Axe], [Objectif principal],
  [*P0*], [Préservation des acquis V3/V5], [Ne rien casser de l'existant déjà validé],
  [*P0*], [Validation GTX 1080], [Conserver la référence de qualité documentée],
  [*P1*], [Sensibilité et précision mixte], [Améliorer le ratio qualité/taille],
  [*P1*], [Cost model], [Mesurer et modéliser les vrais coûts matériels],
  [*P1*], [Working set et I/O], [Optimiser les modèles dépassant la RAM],
  [*P2*], [Profilage MoE dynamique], [Expert ID, traces de routage réel],
  [*P2*], [Cache expert adaptatif], [Réduire drastiquement les lectures SSD],
  [*P2*], [Quantification par expert], [Optimiser finement les poids MoE],
  [*P2*], [Compression physique], [Réduire les transferts sans perte de qualité],
  [*P3*], [Z2/D2 adaptatif], [Automatiser les décisions d'allocation],
  [*P3*], [Backend Qualcomm], [Valider les kernels et les placements réels],
  [*P3*], [Optimisation MTP], [Augmenter le débit de génération],
)

// ============================================================================
// 10. PRINCIPES À NE PAS PERDRE
// ============================================================================
= Principes à ne pas perdre

Les treize règles d'or ci-dessous résument l'expérience accumulée sur les campagnes de mesure GTX 1080, GTX 1080 Ti et Qualcomm :

+ *Ne jamais confondre taille du checkpoint et working set.*
+ *Ne jamais confondre format compact et kernel rapide.*
+ *Ne jamais confondre compatibilité backend et accélération réelle.*
+ *Ne jamais considérer la perplexité comme l'unique mesure de qualité.*
+ *Ne jamais imposer une précision uniforme.*
+ *Ne jamais quantifier tous les experts de la même manière sans mesurer leur sensibilité.*
+ *Ne jamais considérer le cache LRU comme optimal par défaut.*
+ *Toujours séparer les octets demandés, transférés et réellement lus.*
+ *Toujours mesurer le coût de déquantification.*
+ *Toujours confronter les prédictions aux mesures matérielles.*
+ *Préserver les résultats GTX 1080 / 1080 Ti comme références expérimentales.*
+ *Préserver les travaux Qualcomm HTP / Adreno comme axe matériel à part entière.*
+ *Distinguer strictement les fonctionnalités implémentées, validées et encore expérimentales.*

// ============================================================================
// CONCLUSION
// ============================================================================
= Conclusion

Le projet *n'est pas simplement un quantificateur*. C'est une *architecture d'optimisation complète* qui associe :

- Analyse structurelle et numérique ;
- Quantification adaptative de précision mixte ;
- Profilage matériel multi-accélérateurs ;
- Gestion fine de mémoire et working-set contraint ;
- Optimisation des transferts et profilage MoE ;
- Prédiction de performance calibrée ;
- Planification multiobjectif sous contraintes ;
- Validation expérimentale rigoureuse à trois niveaux.

Les travaux sur GTX 1080 ont déjà démontré l'intérêt concret de l'allocation non uniforme de précision. Les travaux sur GTX 1080 Ti ont exploré les limites des représentations extrêmement compactes. Les travaux sur Qualcomm ont prouvé l'importance décisive de mesurer les placements et les fallbacks réels.

*Profiler V3/V5* apporte la base de mesure et de prédiction. *Quantification Algo* apporte la recherche de représentations numériques optimales. *Z2/D2 Planner* apporte la sélection et l'orchestration des stratégies. La finalité est de *trouver automatiquement la meilleure représentation et la meilleure stratégie d'exécution pour un modèle, une qualité cible, un budget mémoire et un matériel donnés*.

Kimi K3 constitue le prochain terrain d'évaluation de cette architecture — il vient la prolonger, et non remplacer les acquis précédents.

// ============================================================================
// ANNEXES
// ============================================================================
= Annexes

== Structure recommandée du dépôt logiciel

```text
quant_project/
├── doc/                        # cette documentation (source Typst + PDF)
├── figures/                    # diagrammes sources (Mermaid) et rendus PNG
├── src/adaptive_quant/         # bibliothèque Python
│   ├── capability_db.py        # profils matériels cibles
│   ├── quant_formats.py        # registre des formats numériques
│   ├── profile_model.py        # cartographie structurelle
│   ├── profiler.py             # profils mémoire & budgets
│   ├── predict_from_hf*.py     # prédiction de performances
│   ├── parse_hexagon.py        # traces Qualcomm HTP
│   ├── sensitivity_engine.py   # sensibilité par tenseur
│   ├── quantization_engine.py  # candidats & reconstruction
│   ├── mixed_precision.py      # allocation optimale de précision
│   ├── expert_profiler.py      # profilage dynamique MoE
│   ├── expert_cache.py         # simulateur de cache d'experts
│   ├── cost_model.py           # modèle de coûts T_token
│   ├── planner.py              # pipeline de décision Z2/D2
│   ├── validation.py           # vérification L1–L3
│   └── cli.py                  # interface en ligne de commande
├── tests/                      # suite de tests automatisés
└── examples/                   # scripts de démonstration (Kimi K3, etc.)
```

== Glossaire des abréviations

#table(
  columns: (1.2fr, 3fr),
  inset: 6pt,
  stroke: 0.4pt + luma(200),
  [Abréviation], [Signification],
  [AWQ], [Activation-aware Weight Quantization],
  [bpw], [Bits per weight (bits effectifs par poids)],
  [GGUF], [Format de sérialisation de modèles (successeur de GGML)],
  [GPTQ], [Generalized Post-Training Quantization for Generative Pre-trained Transformers],
  [HTP], [Hexagon Tensor Processor (NPU Qualcomm)],
  [KL], [Divergence de Kullback-Leibler],
  [MLA], [Multi-head Latent Attention],
  [MoE], [Mixture of Experts],
  [MTP], [Multi-Token Prediction],
  [QAIRT], [Qualcomm AI Runtime],
  [RMS], [Root Mean Square (erreur quadratique moyenne)],
  [SNR], [Signal-to-Noise Ratio (rapport signal sur bruit)],
  [VTCM], [Vector Tightly-Coupled Memory (mémoire locale HTP)],
)
