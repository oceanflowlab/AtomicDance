# Trainingskontext-Variation: Literatur & Werkzeuge

**Grundlagen- und aktuelle Paper/Repos für die nächste Projektphase: den Trainingslauf selbst zur experimentellen Variable machen**

*Kontext: Der erste Trainingslauf (10 h auf 1×A100) hat den „Interpretability Cone" fixiert — alle Befunde in [`mechinterp_planner_report.md`](mechinterp_planner_report.md) gelten für genau diese Gewichte. Mit Spot-Instanzen (8×H100) schrumpft ein Lauf auf Minuten; damit werden Seeds, Datenkomposition, Schedule und Hilfs-Losses zu sweepbaren Variablen, und das Notebook (v5) wird zum standardisierten Assay über viele Läufe. Recherche-Stand: 19. Juli 2026; alle arXiv-IDs von den Recherche-Agenten gegen echte Suchtreffer verifiziert (Ausnahmen markiert).*

---

## 0. Die sechs Leitplanken aus der Literatur

1. **Auf Algorithmus-/Subraum-Ebene vergleichen, nicht auf Komponenten-Ebene.** Circuits bleiben über Training und Skala als *Algorithmus* stabil, während die implementierenden Heads/Neuronen wechseln (Tigges et al. 2024); SAE-Einzelfeatures sind seed-instabil, ihre *Subräume* reproduzieren (arXiv 2606.12138). Für uns: Die Uhr über Läufe hinweg als Subraum (CKA/SVCCA zwischen Uhr-Richtungsräumen) tracken — nicht als „Komponente c63".
2. **Mehrere Seeds pro Bedingung, Verteilungen berichten.** Emergenz ist über Seeds stochastisch (arXiv 2606.25010), Kausal-Scores haben hohe intrinsische Varianz (arXiv 2510.00845), Ausreißer-Seeds sind leicht zu finden (arXiv 2109.08203). Statistik-Blaupause: MultiBERTs-Multi-Bootstrap.
3. **Der nächste Verwandte der Uhr ist publiziert:** „Subliminal Clocks" (arXiv 2607.01774) — latente Denoising-Fortschritts-Repräsentation in Diffusions-LMs. Unsere Abgrenzung: frameweises *Instabilitäts*-Signal (invertierte Polarität, Movement-only-kontrolliert) statt globaler Fortschritts-Uhr.
4. **Emergenz datieren statt nur Endzustand messen:** log-gespacte Checkpoints (Pythia-Schema) + LLC/devinterp als circuit-agnostische Phasendetektion; aus dem fertig analysierten Mechanismus lassen sich Fortschrittsmaße destillieren (Nanda et al. 2023 — bei uns: der Uhr-Gap als Progress Measure).
5. **Fairness der Varianten:** Hyperparameter via µP über Breiten transferieren; Optimizer-Vergleiche nur mit fairem Tuning (Fantastic-Optimizers-Lektion). Sonst misst man Tuning-Artefakte statt Trainingskontext-Effekte.
6. **Bei 19M Parametern ist die GPU nicht der Engpass.** Speedrun-Rezepte (Muon, bf16, compile) plus Dataloader-Denken (FFCV/Composer); Seeds können parallel auf 1–2 GPUs je Lauf laufen statt 8 GPUs je Lauf. Spot-Stack: SkyPilot Managed Jobs → torchrun `--max-restarts` → DCP `async_save` → 2-Minuten-Notice als Checkpoint-Hook.

---

## T1 — Universalität & Seed-/Lauf-Varianz von Mechanismen

### Grundlagen

- **Convergent Learning** (Li, Yosinski et al., 2015) — arXiv:1511.07543 — *Ur-Referenz und Vokabular: individuell-ähnliche vs. subspace-ähnliche Features über unabhängige Läufe.*
- **SVCCA** (Raghu et al., 2017) — arXiv:1706.05806 — [github.com/google/svcca] — *Affin-invarianter Repräsentationsvergleich zwischen Läufen, auch über den Trainingsverlauf.*
- **CKA** (Kornblith et al., 2019) — arXiv:1905.00414 — *Standardmaß für Schicht-Korrespondenz über verschiedene Initialisierungen; das Werkzeug für den Uhr-Subraum-Vergleich.*
- **BERTs of a feather do not generalize together** (McCoy et al., 2019) — arXiv:1911.02969 — *100 identische Läufe, gleiche In-Distribution-Performance, massiv verschiedene Generalisierung — gleiche Loss-Kurve ≠ gleicher Mechanismus.*
- **On the Stability of Fine-tuning BERT** (Mosbach et al., 2020) — arXiv:2006.04884 — [github.com/uds-lsv/bert-stable-fine-tuning] — *Trennt Optimierungs- von Generalisierungs-Varianz — wichtiges Framing für Seed-Sweeps.*
- **The MultiBERTs** (Sellam et al., 2021) — arXiv:2106.16163 — [HF: google/multiberts-seed_0..24] — *25 Seeds + 140 Zwischencheckpoints + Multi-Bootstrap-Statistik — die direkte Design-Blaupause für unseren Seed-Sweep.*
- **Git Re-Basin** (Ainsworth et al., 2022) — arXiv:2209.04836 — [github.com/samuela/git-re-basin] — *Permutations-Symmetrien: „gleich" kann Umsortierung bedeuten — vor jedem naiven Neuron-/Head-Vergleich zu berücksichtigen.*
- **Model Zoos** (Schürholt et al., 2022) — arXiv:2209.14764 — [github.com/ModelZoos/ModelZooDataset] — *Modell-Populationen (Seeds × HPs) als Analyseobjekt — konzeptionelles Vorbild für die geplante Planner-Population.*
- **Mechanistic Mode Connectivity** (Lubana et al., 2023) — arXiv:2211.08422 — *Mechanistische Ähnlichkeit über geteilte Invarianzen; fehlende lineare Konnektivität ⇒ verschiedene Mechanismen — Test, ob Läufe mit/ohne Uhr im selben Becken liegen.*
- **Pythia** (Biderman et al., 2023) — arXiv:2304.01373 — [github.com/EleutherAI/pythia] — *Referenz-Design für Checkpoint-Suiten (log-gespaced früh); Vorlage für unser Checkpoint-Schema.*
- **OLMo** (Groeneveld et al., 2024) — arXiv:2402.00838 — [github.com/allenai/OLMo] — *Volltransparente Trainingsläufe (Checkpoints, Logs, Daten) — Muster für veröffentlichungsfähige Varianten-Buchführung.*
- **Universal Neurons in GPT-2** (Gurnee et al., 2024) — arXiv:2401.12181 — [github.com/wesg52/universal-neurons] — *Operationale Universalitäts-Metrik (Aktivierungs-Korrelation über Seed-Läufe; nur 1–5 % universell) — direkt auf Uhr-Richtungen anwendbar.*
- **The Platonic Representation Hypothesis** (Huh et al., 2024) — arXiv:2405.07987 — [github.com/minyoungg/platonic-rep] — *Konvergenz-These über Skala/Datenvielfalt — Erwartungsrahmen, welche Mechanismen gegen Datenvariation stabil sein sollten.*

### Aktuell (2024–2026)

- **LLM Circuit Analyses Are Consistent Across Training and Scale** (Tigges et al., 2024) — arXiv:2407.10827 — *Leit-Hypothese: Algorithmus stabil, Komponenten wechseln — exakt unser 17b/18c-Muster (Head-Ebene leer, Richtungs-Ensemble konzentriert).*
- **Quantifying Feature Space Universality via SAEs** (Lan et al., 2024) — arXiv:2410.06981 — *Cross-Model-Feature-Matching (Korrelation + SVCCA) — Methodik für SPD-Befund-Vergleiche über Läufe.*
- **SAEs Trained on the Same Data Learn Different Features** (Paulo & Belrose, 2025) — arXiv:2501.16615 — *Warnung: gelernte Zerlegungs-Richtungen sind seed-instabil — gilt analog für unsere SPD-Komponenten.*
- **Universal Sparse Autoencoders** (Thasarathan et al., 2025) — arXiv:2502.03714 — *Gemeinsamer Konzeptraum über mehrere Modelle — Werkzeug zum Alignieren von Bewegungs-Konzepten über Planner-Varianten.*
- **Toward a Theory of Generalizability in MI Research** (Trott, 2025) — arXiv:2509.22831 — *Fünf Korrespondenz-Achsen (funktional/entwicklungs-/positional/relational/konfigurational) — fertiges Klassifikationsraster für „wie universell ist die Uhr?".*
- **MI as Statistical Estimation: A Variance Analysis** (Méloux et al., 2025) — arXiv:2510.00845 — *Hohe intrinsische Varianz von Kausal-Scores — Pflicht: Stabilitätsmetriken über Seeds berichten.*
- **Universal Neurons in GPT-2: Emergence, Persistence, Impact** (2025) — arXiv:2508.00903 — *Wann universelle Neuronen entstehen und persistieren — Vorlage für die Entwicklungs-Achse.*
- **Unstable Features, Reproducible Subspaces** (Gerasimov et al., 2026) — arXiv:2606.12138 — *Einzelfeatures instabil, Subräume konsistent — das Argument, die Uhr auf Subraum-Ebene zu vergleichen (Leitplanke 1).*

---

## T2 — Trainingsdynamik & Emergenz von Circuits

### Grundlagen

- **In-context Learning and Induction Heads** (Olsson et al., 2022) — arXiv:2209.11895 — *Prototyp eines checkpoint-datierten Mechanismus-Entstehens (Phasenübergang mit Loss-Bump).*
- **Grokking** (Power et al., 2022) — arXiv:2201.02177 — *Verzögerte Mechanismus-Emergenz nach dem Loss-Plateau — Nullhypothese fürs Uhr-Timing.*
- **Progress Measures for Grokking** (Nanda et al., 2023) — arXiv:2301.05217 — [github.com/mechanistic-interpretability-grokking/progress-measures-paper] — *Aus fertigem Circuit kontinuierliche Fortschrittsmaße destillieren — Rezept: der Uhr-Gap als Progress Measure über Checkpoints.*
- **Explaining Grokking Through Circuit Efficiency** (Varma et al., 2023) — arXiv:2309.02390 — *„Ungrokking/Semi-Grokking" als Circuit-Wettbewerb — Rahmen für Uhr-vs-Shortcut-Varianten.*
- **Sudden Drops in the Loss** (Chen et al., 2023) — arXiv:2309.07311 — *Loss-Events kausal mit interner Strukturbildung verbunden — Blaupause fürs Alignieren von Loss-Kurve und Uhr-Emergenz.*
- **Local Learning Coefficient (SLT)** (Lau et al., 2023) — arXiv:2308.12108 — [github.com/timaeus-research/devinterp] — *Circuit-agnostische Phasendetektion über das Training — ideal für Minuten-Läufe mit vielen Checkpoints.*
- **The Developmental Landscape of In-Context Learning** (Hoogland et al., 2024) — arXiv:2402.02364 — *Diskrete Entwicklungsstadien, detektierbar via LLC/Essential Dynamics.*
- **Refined LLCs: Attention-Head-Spezialisierung** (Wang et al., 2024) — arXiv:2410.02984 — [github.com/timaeus-research/paper-rllcs-2024] — *Komponentenweise LLCs datieren Spezialisierung — übertragbar auf die L4-MLPs der Uhr.*
- **The Quantization Model of Neural Scaling** (Michaud et al., 2023) — arXiv:2303.13506 — [github.com/ejmichaud/quantization-model] — *Fähigkeiten als Quanta in datenhäufigkeitsbestimmter Reihenfolge — Vorhersage: Uhr-Erwerb hängt an der Häufigkeit instabiler Frames.*
- **Lottery Tickets** (Frankle & Carbin, 2019) — arXiv:1803.03635 — *Initialisierungs-Subnetze — Hintergrund zur Frage, ob die Uhr je Seed im „gleichen" Subnetz entsteht.*
- **The Transient Nature of Emergent In-Context Learning** (Singh et al., 2023) — arXiv:2311.08360 — *Mechanismen können nach Emergenz wieder verschwinden — Läufe über den Emergenzpunkt hinaus fortsetzen.*
- **Competition Dynamics Shape Algorithmic Phases** (Park et al., 2024) — arXiv:2412.01003 — *Trainingsverteilungs-Parameter bestimmen den Gewinner-Algorithmus — genau unser Design „Kontext variieren, Mechanismus messen".*

### Aktuell (2025–2026)

- **Subliminal Clocks: Latent Time Modelling in Diffusion LMs** (Rulli et al., 2026) — arXiv:2607.01774 — *Der nächste Verwandte der latenten Uhr (Denoising-Fortschritt in Diffusions-LMs, extrahier- und steuerbar) — Pflichtlektüre für die Abgrenzung; Grundlage der subliminal-coax-Notebooks.*
- **Natural Ungrokking** (2026) — arXiv:2606.26050 — *Regel emergiert und kollabiert im Pretraining, gesteuert über Support-Frequenz (Kill/Rescue-Asymmetrie) — Muster-Paper für „Uhr-Überleben" unter Datenvarianten (liegt als Lernmappe im Projekt vor).*
- **Position: A Science of AI Must Study Training Dynamics** (Biderman et al., 2026) — arXiv:2606.06533 — *Programmatische Rechtfertigung des gesamten Vorhabens.*
- **Crosscoding Through Time** (Bayazit et al., 2025) — arXiv:2509.05291 — *Crosscoder alignieren Features über Checkpoints; RelIE datiert kausale Relevanz — beste aktuelle Werkzeugkette für Uhr-Tracking über Checkpoints.*
- **Evolution of Concepts in LM Pre-Training** (Ge et al., 2025) — arXiv:2509.17196 — *Zwei Phasen (statistisch → Feature-Lernen) — wann im Training ein Signal wie die Uhr zu erwarten ist.*
- **Beyond Induction Heads: Multi-Phase Circuit Emergence** (2025) — arXiv:2505.16694 — *Zwischenlösungen („Proto-Circuits") — Hypothese einer Proto-Uhr.*
- **ICL Strategies Emerge Rationally** (Wurgaft et al., 2025) — arXiv:2506.17859 — *Loss-Komplexitäts-Tradeoff sagt Strategie-Wechsel voraus — Kandidat für ein Vorhersagegesetz der Uhr-Emergenz.*
- **Predicting the Emergence of Induction Heads** (2025) — arXiv:2511.16893 — *Emergenzpunkt aus Batch-/Kontextgröße vorhersagbar — Vorbild für ein „Uhr-Emergenz-Gesetz" über HP-Varianten.*
- **Emergent Capabilities Arise Randomly** (Baherwani et al., 2026) — arXiv:2606.25010 — *Emergenz seed-stochastisch — Begründung für n≥5 Seeds pro Bedingung.*
- **Mechanism Shift: AR → Masked Diffusion** (Kong et al., 2026) — arXiv:2601.14758 — *Lokale Circuits vererben sich, globale werden neu gebaut — welche Uhr-Anteile sind D3PM-spezifisch?*
- **SLT-Perspektive auf Grokking/Phasenübergänge** (2025) — arXiv:2512.00686 — *Theoretischer Unterbau, falls Uhr-Emergenz als LLC-Sprung sichtbar wird.*
- **Review: Developmental Interpretability in LLMs** (Kendiukhov, 2025) — arXiv:2508.15841 — *Feld-Überblick und Zitatquelle.*
- *Randfunde (checkpoint-freie Diagnostik über Gewichtsspektren):* arXiv:2602.02859 (Anti-Grokking via WeightWatcher), arXiv:2506.04434 (HTSR-Grokking).

---

## T3 — Trainingsdaten-/Kontext-Attribution

- **Datamodels** (Ilyas et al., 2022) — arXiv:2202.00622 — [github.com/MadryLab/datamodels] — *Die „trainiere-viele-Male"-Logik als Messinstrument: Verhalten aus Datenkomposition vorhersagen — theoretisches Fundament der Genre-Ablationen.*
- **TRAK** (Park et al., 2023) — arXiv:2303.14186 — [github.com/MadryLab/trak] — *Datamodel-Qualität aus wenigen Modellen — praktikable Attribution pro Variante.*
- **Influence Functions at Scale (EK-FAC)** (Grosse et al., 2023) — arXiv:2308.03296 — *„Welche Daten lehrten Mechanismus X" — Vorlage für Uhr-/Beat-Attribution.*
- **Data Shapley** (Ghorbani & Zou, 2019) — arXiv:1904.02868 — [github.com/amiratag/DataShapley] — *Spieltheoretische Datenbewertung via Retraining-Subsets — anschlussfähig an billige Reruns.*
- **TDA-Survey** (Hammoudeh & Lowd, 2022) — arXiv:2212.04612 — *Methoden-Landkarte zur Auswahl.*
- **Simfluence** (Guu et al., 2023) — arXiv:2303.08114 — [github.com/google-research/simfluence] — *Trainings-Läufe als Simulator; fängt Reihenfolge-/Curriculum-Effekte — passgenau für Schedule-Varianten.*
- **In-Run Data Shapley** (Wang et al., 2024) — arXiv:2406.11011 — *Attribution ohne Retraining in einem Lauf — pro Rerun direkt messbar, welche Daten die Mechanismen treiben.*
- **Mechanistic Data Attribution** (Chen et al., 2026) — arXiv:2601.21996 — *Interpretierbare Einheiten per Influence auf Trainingssamples zurückführen; strukturierte Daten als „mechanistischer Katalysator" — unmittelbares Vorbild für die Uhr.*
- **ExPLAIND** (2025) — arXiv:2505.20076 — *Modell-, Daten- und Trainingsverlaufs-Attribution vereint — für gemeinsame Zerlegung von Schedule-/Loss-/Daten-Effekten.*
- **Influence Functions for Diffusion Models** (Mlodozeniec et al., 2024) — arXiv:2410.13850 — *Skalierbare IF-Attribution explizit für Diffusion.*
- **D-TRAK** (Zheng et al., 2023) — arXiv:2311.00500 — [github.com/sail-sg/D-TRAK] — *TRAK für Diffusion, robust über Checkpoints/Timesteps — direkt anwendbares Rezept.*

---

## T4 — Diffusion & Motion: Substrat, Interpretierbarkeit, Beat-Losses

### Diffusion

- **D3PM** (Austin et al., 2021) — arXiv:2107.03006 — *Das Substrat des Planners; Referenz für Schedule-/Übergangskern-Varianten (uniform vs. absorbing!).*
- **Geometry-Adaptive Harmonic Representations** (Kadkhodaie et al., 2024) — arXiv:2310.02557 — *Konvergenz zweier Netze zur selben Score-Funktion ab genug Daten — Anker für Datenmengen-Ablationen.*
- **Critical Windows** (Li & Chen, 2024) — arXiv:2403.01633 — *Enge Sampling-Zeitfenster der Merkmals-Entstehung — konzeptuell verwandt mit der Uhr als Timing-Signal.*
- **Emergence and Evolution of Interpretable Concepts in Diffusion Models** (2025) — arXiv:2504.15473 — *Konzept-Emergenz über das Training in Diffusion — Methodik für unsere Reruns.*
- **SAEs für Text-to-Image-Diffusion** (2024) — arXiv:2410.22366 — *Kausal wirksame Features + Block-Spezialisierung in Diffusion.*
- **Mechanistic Interpretability of Diffusion Models** (2025) — arXiv:2506.17237 — *Circuit-Level-Interventionen mit kausaler Validierung in Diffusion — Vorbild für Uhr-/Integrator-Patching.*
- **DifFRACT** (2025/26) — arXiv:2606.15796 — *SAE-Features + Circuit-Tracing in Diffusion — aktuellstes Werkzeug für Feature→Circuit-Pfade.*
- **Why Diffusion Models Don't Memorize** (Bonnaire et al., 2025) — *arXiv-ID unverifiziert (Suchhinweis: Titel + „implicit dynamical regularization") — wachsendes Fenster zwischen Generalisierungs- und Memorisierungs-Onset — relevant für Trainingslängen-Effekte.*

### Motion / Musik→Tanz (inkl. Beat-Loss-Vorlagen)

- **AI Choreographer / FACT + AIST++** (Li et al., 2021) — arXiv:2101.08779 — [github.com/google-research/mint] — *Unser Datensatz (10 Genres) — Basis der genre-weisen Ablationen.*
- **Bailando** (Siyao et al., 2022) — arXiv:2203.13055 — [github.com/lisiyao21/Bailando] — *Beat-Align-Reward via Actor-Critic — Vorbild für einen nachgelagerten Takt-Anreiz.*
- **EDGE** (Tseng et al., 2023) — arXiv:2211.10658 — [github.com/Stanford-TML/EDGE] — *Diffusion mit Auxiliary-Loss (Contact Consistency) + Beat-Alignment-Metrik — Konstruktionsmuster für den Beat-Phasen-Hilfs-Loss.*
- **MDM** (Tevet et al., 2023) — arXiv:2209.14916 — [github.com/GuyTevet/motion-diffusion-model] — *Sample-Prädiktion ermöglicht geometrische Hilfs-Losses — das Pattern für Phase-Targets im D3PM.*
- **Beat-It** (Huang et al., 2024) — arXiv:2407.07554 — *Expliziter Beat-Alignment-Loss (nearest-beat-distance) + Beat-Hit/Coverage-Metriken — die konkreteste Vorlage für Design UND Evaluation des Hilfs-Losses.*
- **Lodge** (Li et al., 2024) — arXiv:2403.10518 — [github.com/li-ronghui/LODGE] — *Tanz-Primitive als Zwischenrepräsentation + Refine-Hilfs-Losses — strukturierte Trainingsziele als Kontext-Variante.*

---

## T5 — Schnelltraining & Spot-Infrastruktur

### Speedrun-Rezepte

- **modded-nanogpt** (Keller Jordan et al.) — github.com/KellerJordan/modded-nanogpt — *Der maßgebliche 8×H100-Speedrun (124M-GPT-2 in ~90 s, 21+ dokumentierte Rekorde) — exakt unsere Zielhardware; Worklog: tylerromero.com/posts/nanogpt-speedrun-worklog.*
- **Muon** — github.com/KellerJordan/Muon — *Orthogonalisierender Momentum-Optimizer (+~35 % Speed) — billigster Einzelhebel für den D3PM-Trainer.*
- **cifar10-airbench** — github.com/KellerJordan/cifar10-airbench / arXiv:2404.00498 — *Das Paradigma „Sekunden-Training ⇒ Hunderte Läufe pro Experiment" explizit begründet.*
- **Cramming** (Geiping & Goldstein, 2022) — arXiv:2212.14034 — [github.com/JonasGeiping/cramming] — *Pipeline-Ablationen unter hartem Compute-Budget — welche Tricks beim Herunterskalieren wirklich tragen.*
- **llm.c** (Karpathy) — github.com/karpathy/llm.c — *~60 % MFU auf einem 8-GPU-Node als Messlatte.*
- **nanochat** (Karpathy, 2025) — github.com/karpathy/nanochat — *Reproduzierbare One-Node-Runs inkl. Kostenrahmen.*
- **TinyStories** (Eldan & Li, 2023) — arXiv:2305.07759 — *Kleine Modelle lernen kohärent bei passender Datenverteilung — 19M dürfen klein bleiben; die Daten sind der Sweep-Parameter.*
- **MosaicBERT + Composer** — arXiv:2312.17482 / github.com/mosaicml/composer — *Gestapelte Recipe-Tricks als komponierbare Trainer-Bibliothek.*
- **µP / mup** (Yang et al., 2022) — arXiv:2203.03466 — [github.com/microsoft/mup] — *HP-Transfer über Breiten (µTransfer); Weiterentwicklung u-µP: arXiv:2407.17465 — zentral für faire Varianten (Leitplanke 5).*
- **torchtitan** — github.com/pytorch/torchtitan / arXiv:2410.06511 — *PyTorch-natives Pretraining-Gerüst inkl. async Distributed Checkpointing.*
- **Fantastic Pretraining Optimizers I+II** (2025/26) — arXiv:2509.02046, arXiv:2606.16899 — *Faire Optimizer-Sweeps: Muon ~1,4× über AdamW; viele 2×-Claims verschwinden bei fairem Tuning.*
- **Automated LLM Speedrunning Benchmark** (2025) — arXiv:2506.22419 — *Alle NanoGPT-Rekorde als reproduzierbare Diffs — strukturierte Zutatenliste.*
- **Small-scale Proxies for Training Instabilities** (Wortsman et al., 2023) — arXiv:2309.14322 — *Instabilitäten gezielt klein reproduzieren — für aggressive LRs im Minuten-Regime.*

### Spot-/Preemption-Stack (konkret für AWS 8×H100)

- **SkyPilot** — github.com/skypilot-org/skypilot — *`sky jobs launch --use-spot`: Preemption-Erkennung, Ersatz-Provisionierung (Region-/Cloud-Wechsel), Auto-Resume vom Bucket-Checkpoint — das Frontend für die Varianten-Job-Queue.*
- **„Can't Be Late" + spot-traces** (NSDI '24 Best Paper) — usenix.org/conference/nsdi24/presentation/wu-zhanghao / github.com/skypilot-org/spot-traces — *Spot-unter-Deadline-Policy (27–84 % Ersparnis) + echte AWS-Verfügbarkeits-Traces für die Planung.*
- **AWS Spot Best Practices** — docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html — *2-Minuten-Notice + Rebalance-Recommendation als Sofort-Checkpoint-Trigger; `price-capacity-optimized`; p5-Spot ist knapp → Multi-AZ + p4d-Fallback.*
- **torchrun elastic** — docs.pytorch.org/docs/stable/elastic/run.html — *`--max-restarts`-Pattern: Prozess-Restart + Snapshot-Load — Minimal-Fehlertoleranz für jedes Trainingsskript.*
- **DCP async_save** — docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html — *Asynchrones Checkpointing; bei 19M kostet ein Checkpoint/Minute praktisch nichts → Preemption-Verlust in Sekunden.*
- **torchft** (Meta, 2025) — github.com/meta-pytorch/torchft — *Per-Step-Fehlertoleranz ohne Stop-the-World — nur nötig, falls später multi-node-elastisch.*
- **Ray Train/Tune** — docs.ray.io — *Alternative: fehlertolerante Trial-Queue für Seeds-/HP-Sweeps.*
- **HF accelerate** — github.com/huggingface/accelerate — *Gleiches Skript auf 1 GPU (Debug) und 8×H100 (Sweep).*
- **Marin Open Lab** (Stanford CRFM, 2025/26) — github.com/marin-community/marin — *Jedes Experiment als versionierter, reproduzierbarer Schritt (inkl. Fehlschläge) — Buchführungs-Vorbild für die Läufe-Population.*
- **FFCV** — github.com/libffcv/ffcv / arXiv:2306.12517 — *Dataloading als Engpass-Löser — die Datamodels-Infrastruktur (Hunderttausende Läufe) als Machbarkeits-Beweis.*
- **torch.manual_seed(3407) is all you need** (Picard, 2021) — arXiv:2109.08203 — *Seed-Scan-Warnung: nie auf Einzelläufen gründen.*

---

## Versuchsplan-Skizze: Achsen × Assay

**Achsen des Trainingskontexts** (jeweils n ≥ 5 Seeds; HPs via µP fixiert):

| Achse | Varianten | Erwartung / Frage |
|---|---|---|
| A. Seed | 5–25 Seeds pro Bedingung | Universalität der Uhr (Subraum-CKA), Varianz aller Assay-Metriken (Multi-Bootstrap) |
| B. Datenkomposition | Genre-Ablationen; Songs/Genre ↑ (aus Train-Split) | Quantization-Model-Vorhersage; löst nebenbei Artefakt A1 (Genre-Probe wird messbar) |
| C. Diffusions-Schedule | cosine/linear; T = 50/100/200; uniform vs. absorbing Kern | Hängt die Instabilitäts-Uhr am Schedule? (D3PM-Kern-Variante = stärkster Eingriff) |
| D. Hilfs-Loss | Beat-Phase-Loss (à la Beat-It/EDGE/MDM), Gewicht gestuft | **Die kausale Intervention zu Befund 1:** Entsteht ein Takt-Integrator, und verändert er die Uhr? |
| E. `--transition-weight` | bereits vorhandenes Trainings-Flag, gestuft | Direkter Hebel gegen den Transition-Kollaps (`transition_frac` 0.77–1.0) — verschiebt er den A4-Confound? |
| F. Architektur | Tiefe 4/8/12, Breite via µP | Wandert der Uhr-Ort (L4.linear2 ≈ „letzte Station vor Readout")? |
| G. FF-Aktivierung | `--activation gelu/poly2/square/gaussian` (seit v10; `poly2` = lernbare Parabel pro FF-Kanal nach [LifeNNgine](https://github.com/Erikiss/LifeNNgine)-Vorbild) | **Die kausale Intervention zu Befund 4/7/8:** Band-selektive Automaten-Regeln werden mit parabolischer Aktivierung dramatisch leichter lernbar (GoL: PolyKAN L(1,1) genügt, wo ReLU scheitert). Vorhersage: Uhr wird kompakt (SPD-Konzentration ↑), gewichts-lesbar (21er-Extraction besteht) und geometrisch schärfer (23er-Parabel) — Superposition/Gating als Kosten der monotonen GELU-Basis, nicht als Design |

**Assay pro Lauf** (Notebook v5 automatisiert + Checkpoint-Schema log-gespaced à la Pythia):

1. Uhr-Gap (16b, movement-only) — die Kernmetrik, auch als Progress Measure über Checkpoints (Nanda-Rezept).
2. SPD-Konzentration (Top-10-Anteil, Layer-Verteilung) — Vergleich über Läufe auf **Subraum-Ebene** (CKA der Top-Schadens-Richtungen), nicht auf Komponenten-IDs.
3. Geometrie-Dissoziation (19c-Rotation: Grammatik-Invarianz vs. Uhr-Kollaps).
4. Beat-Phase-R² (15b) + Transition-Beat-Alignment (15d) — Erfolgskontrolle für Achse D.
5. `transition_frac` vs. Ground-Truth-Rate + BAS (Verhaltens-Gesundheit).
6. LLC über Checkpoints (devinterp) — Emergenz-Timing, circuit-agnostisch.
7. Parabel-Geometrie (23c/23d: EV Top-2, Parabel-R² gegen Phasen-Surrogate) + Steering-Verdikt (24c) — der Prozess-Geometrie-Assay; zusammen mit SPD-Konzentration und 21er-Extraction die Erfolgskontrolle für Achse G.

---

## Priorisierte Leseliste (Top 10 für den Einstieg)

1. Subliminal Clocks (2607.01774) — Abgrenzung der Uhr.
2. Tigges et al. (2407.10827) — Vergleichsebene wählen.
3. MultiBERTs (2106.16163) — Sweep-Design + Statistik.
4. Unstable Features, Reproducible Subspaces (2606.12138) — Subraum-Vergleich.
5. Natural Ungrokking (2606.26050) — Überlebens-Dynamik unter Datenvarianten.
6. Mechanistic Data Attribution (2601.21996) — Mechanismus ↔ Daten.
7. Beat-It (2407.07554) — Hilfs-Loss-Konstruktion.
8. Nanda et al. (2301.05217) — Uhr-Gap als Progress Measure.
9. modded-nanogpt + µP (Repo + 2203.03466) — Trainings-Setup.
10. Méloux et al. (2510.00845) — Varianz-Reporting-Pflicht.

*Erstellt aus vier parallelen Recherche-Läufen (Universalität/Seed-Varianz · Trainingsdynamik/Emergenz · Attribution/Diffusion/Motion · Schnelltraining/Infra); Duplikate zusammengeführt. Nicht verifizierte Angaben sind markiert.*
