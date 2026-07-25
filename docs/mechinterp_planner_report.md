# Die latente Uhr des AtomicDance-Planners

**Eine mechanistische Untersuchung in acht Methoden — und die Artefakte, die fast zu Fehlschlüssen geführt hätten**

*Notebook: [`notebooks/AtomicDance_MechInterp_Planner.ipynb`](../notebooks/AtomicDance_MechInterp_Planner.ipynb) (v10.1) · Stand: 20. Juli 2026*

---

## Zusammenfassung

Der AtomicDance-Planner — ein selbst trainiertes diskretes Diffusionsmodell (UniformD3PM, 19 Mio. Parameter), das aus 35-dimensionalen AIST++-Musikfeatures frameweise Atomic-Move-Pläne generiert — wurde mit acht unabhängigen Mech-Interp-Methoden untersucht: lineare Probes, nichtlineare MLP-Probes, symbolisches Attention-Programm-Replacement, Stochastic Parameter Decomposition (SPD), Jacobian-Lens mit Geometrie-Variation, eine NLA-analoge Sub-Workspace-Dekodierung, gewichts-basierte Secret Extraction und Trajektorien-Geometrie mit Surrogat-Null (Parabel-Detektion). Das Ergebnis ist eine kohärente mechanistische Geschichte:

1. **Der Planner integriert keinen musikalischen Takt.** Beats werden als lokale Impulse perfekt durchgereicht, aber es existiert keine Beat-Phasen-Repräsentation — weder linear noch nichtlinear dekodierbar, und das Verhalten (Transition↔Beat-Alignment) bestätigt es.
2. **Der Planner führt eine echte, frameweise Verifikations-Spur** („Ist dieses Frame schon final?"), die deutlich mehr ist als Label-Durchleitung: +0.23 AUC über der Label-Identitäts-Baseline auf Movement-Frames, über die gesamte Denoising-Trajektorie.
3. **Diese Spur lebt nicht in einzelnen Attention-Heads** (maximaler Einzel-Head-Schaden: −0.014 von +0.31), sondern ist ein **superponiertes Ensemble von Rang-1-Richtungen**, konzentriert in der MLP-Ausgabematrix von Layer 4 (55 % des Ablationsschadens in Layer 4, 69 % in den `linear2`-Matrizen).
4. **Die Polarität ist invertiert:** Die tragenden Komponenten feuern auf *instabile* Frames — die „Uhr" ist physisch ein **Instabilitäts-Detektor**; Settledness wird als Abwesenheit dieses Signals gelesen. Für einen Denoiser ist das die funktional richtige Kodierung („wo ist noch Arbeit?").
5. **Die Uhr ist ein geometrischer Code, die Plan-Grammatik nicht:** Unter kontrollierten Geometrie-Eingriffen (Skalierung, Rotation im Top-Jacobian-Subraum, Translationen) bleibt der Plan-Inhalt praktisch vollständig invariant (Argmax-Agreement ≥ 0.999 bei moderaten Eingriffen), während die Uhr-Anzeige mit Orientierung und Amplitude kovariiert — bei 90°-Rotation in der Top-2-J-Ebene kollabiert sie um −0.52 bei noch 91 % intakter Grammatik.
6. **Unterhalb des Workspace: getragen, still, lesbar.** ~88 % der Masse typischer Konzept-Richtungen liegt außerhalb des Top-64-J-Zeilenraums. Injiziert man diese Sub-J-Komponenten, bleibt der Output vollständig blind (Plan-Agreement ≥ 0.999, KL ≤ 0.054) — ein extern trainierter Aktivierungs-Dekoder benennt sie dennoch **perfekt** (Accuracy 1.00 bei 9 von 10 Konzepten, **FPR = 0.000**). Dazu eine Schreib-/Lese-Asymmetrie der Uhr: Die SPD-Schreib-Richtung c63 liegt zu 44,8 % im Workspace (3,6× Zufall), die Probe-Lese-Richtung nur zu 5,0 % (0,4× Zufall) — **geschrieben wird im Workspace, gelesen darunter.**
7. **Die Trigger-Landschaft ist strukturiert und kausal — aber die Uhr ist kein naiver Gewichts-Pegel.** Input-Optimierung auf den Uhr-Schreib-Pegel des L4-MLP findet nur **4 breite Attraktoren** (Halo-Abfall ~0.7 %) aus ~120 Starts, alle von Neuron-Seeds/Kompositionen gewonnen und im Embedding-Umfeld der dominanten Movement-Labels verankert (Stimuli *und* Lens-dekodierte Writes kreisen um Label 67/86/34). Die Top-Trigger-Richtung ist kausal potent (Uhr-Shift −1.93 vs. Zufall +0.33 ± 0.81). **Aber:** Als Instabilitäts-Detektor auf echten Frames erreicht der signierte Schreibpegel nur AUC 0.304/0.242 — unter Chance, vorzeichen-korrigiert 0.70–0.76 — weit unter der Probe (0.894). Die Rang-1-Richtungen sind vorzeichen-ambig und **CI-gegated** (input-abhängig aktiviert): Ohne das Gate ist die Uhr aus den Gewichten allein nicht lesbar (A9).

8. **Die Zeitgeometrie ist eine echte Parabel — die lesbare Schritt-Uhr ist Oberfläche.** Ohne injizierte Zeit (t ≡ 99-Replays der echten Sampling-y_t) wächst die Fortschritts-Lesbarkeit mit der Tiefe, bleibt aber mit R² ≤ 0.68 (Top-2: ≤ 0.80) unter der Label-Histogramm-Baseline von y_t (0.89); mit injizierter Zeit repliziert sie die LLaDA/Dream-Referenz trivial (0.97–0.999 in jeder Schicht; A10). Die Schritt-Mittel-Trajektorie dagegen ist extrem niedrigdimensional (EV Top-2 ≥ 0.9986), perfekt geordnet (|Spearman(PC1, τ)| ≥ 0.9997) und tatsächlich parabolisch: Parabel-R² bis 0.993, **signifikant über spektrum-gematchten Surrogaten** in den Layern 1–2 und 6–7 (p = 0.0025 < Bonferroni 0.0063) — obwohl die Surrogate selbst im Mittel R² 0.93 liefern (A11): Ohne Null-Test wäre die Parabel in allen 16 Bedingungs-Layer-Kombinationen „detektiert" worden. Kreuz-konsistent über Musik-Hälften erst in der Tiefe (0.91, Layer 6).

9. **Die Uhr lässt sich steuern — als Ebene, nicht als Fahrplan.** Injektion entlang der lokalen Kurven-Tangente in Layer 6 verschiebt die Verhaltens-Schärfe (Logit-Entropie) früh und mittig perfekt dosis-monoton und vorzeichen-richtig (Spearman −1.00 bei τ ≈ 0.2 und 0.4; rückwärts fast 1:1: α = −4 → −3.8 effektive Schritte), 12× stärker als Zufallsrichtungen (z = 10.3) und bei vollständigem Grammatik-Erhalt (kein einziger Argmax-Flip bis |α| = 4). Aber die Kurven-**Normale** wirkt 1.5× stärker als die Tangente: Kausal ist die **2D-Kurvenebene**, nicht die Tangentenrichtung — These 4 der Vorlage hält in der mittleren Lesart. Der Lauf legte zudem A12 frei: Die vorregistrierte Hold-Rate war durch den Transition-Kollaps ab Schritt 1 gesättigt und hätte ohne kontinuierlichen Zweitkanal ein falsches Null-Verdikt produziert.

Ebenso wichtig wie die Befunde sind die **zwölf methodischen Artefakte**, die unterwegs identifiziert und neutralisiert wurden — jedes davon hätte unentdeckt zu einem plausiblen, aber falschen Schluss geführt.

---

## 1. Untersuchungsobjekt

| | |
|---|---|
| Modell | `UniformD3PM(AtomicPlannerTransformer)` — D3PM mit uniformem Übergangskern |
| Kern | 8-Layer-TransformerEncoder, d=512, 8 Heads, `batch_first=True`, `norm_first=True` |
| Parameter | 19.044.965 trainierbar (+ 77.000 Buffer: `pe [150,1,512]`, `alphas`, `alpha_bars`) |
| Input | 35-dim Musikfeatures pro Frame: `[Envelope \| 20×MFCC \| 12×Chroma \| Peak \| Beat]` |
| Output | Frameweise Logits über 101 Klassen (Label 0 = Transition, 1–100 = Atomic Moves) |
| Diffusion | 100 Schritte, Cosine-Schedule; Sampler resampelt jede Position jeden Schritt |
| Checkpoint | `planner_step22180.pt` (nur lesend analysiert; Modell zu jeder Zeit eingefroren) |
| Daten | AIST++-Test-Split: 372 Fenster à 150 Frames — **nur sBM, genau 1 Song pro Genre** |

Analysezustand der meisten Probes: `t = 99`, festes Uniform-Rausch-y — das Modell sieht effektiv nur die Musik. Trajektorien-Analysen loggen echte Sampling-Zustände (y_t, t) über alle 100 Reverse-Schritte.

---

## 2. Methodische Artefakte — was fast zu Fehlschlüssen geführt hätte

Diese Liste ist das methodische Rückgrat der Arbeit. Jeder Punkt ist ein real aufgetretener Fall.

**A1 — Accuracy 0.0 ist kein Separierbarkeits-Befund.** Die Genre-/Tempo-Probes lieferten exakt 0.0 über alle Layer. Ohne Information läge eine Probe bei Chance (10 %/17 %), nicht bei null. Ursache: Im Test-Split hat jedes Genre genau einen Song → Musik-ID ≡ Genre → der Group-Split hält ganze *Genres* aus dem Training, und eine im Training fehlende Klasse ist für logistische Regression prinzipiell unvorhersagbar. Die 0.0 war strukturell erzwungen. *(Eine automatisch generierte Folie hatte daraus „fehlende lineare Separierbarkeit, Pipeline artefaktfrei" gemacht — das Gegenteil der korrekten Lesart.)*

**A2 — Leakage durch geteilte Songs.** Probes mit Musik-only-Input müssen nach **Musik-ID** gruppiert werden, nicht nach Aufnahme (`base_seq`): Derselbe Song wird von mehreren Tänzern getanzt; sonst liegt identischer Input in Train und Test.

**A3 — Song-konstante Targets sind mit Song-Splits unlösbar.** Tempoklasse und Inter-Beat-Intervall sind innerhalb eines Songs (quasi) konstant — mit Musik-ID-Split bleibt nur Extrapolation (negative R² sogar auf Roh-Input). Lösung: **zeitvariable Targets** (Beat-Phase, Frames-bis-Beat), die in jedem Song den vollen Wertebereich durchlaufen (JoruriPuppet-Prinzip „Beat-Tabelle").

**A4 — Der Label-Identitäts-Confound der Settledness.** Gesampelte Pläne sind transition-lastig (76,5 % Label 0). Damit gilt „settled ≈ aktuelles Label ist 0", trivial aus dem Label-Embedding lesbar. Kontrolle: Label-One-Hot-Baseline + Movement-only-Auswertung. Ergebnis: Auf allen Frames erklärt der Confound fast alles (+0.055 Rest), auf Movement-Frames nicht (+0.23) — die Spur ist echt, aber nur die kontrollierte Metrik zeigt es.

**A5 — Readout-Position begrenzt Kausal-Tests.** Interventionen stromabwärts des Probe-Layers können die Probe-Anzeige nicht beeinflussen: Head-Ablationen in Layer 5–7 zeigten exakt Δ=0 bei Layer-4-Readout — ein Konstruktions-Artefakt, kein Befund. Die `label_boundaries`-Heads der späten Layer sind damit **ungetestet**, nicht entlastet.

**A6 — Loss-Koeffizienten sind domänenspezifisch.** Die SPD-Defaults sind auf LM-KL-Größenordnungen (~1) kalibriert; die Planner-KL liegt bei ~0.004. Importance-Minimality erdrückte die Rekonstruktion (L0: 40 → ~1 in 250 Schritten), die Komponenten trugen nichts (`ci-masked ≈ zero-masked`) — die erste Attribution war uninformativ. Nach Rebalancierung (coeff_imp ÷10, coeff_stoch ×25): 75,9 % Rekonstruktion, valide Attribution. **Qualitäts-Gate:** Komponenten müssen den MLP-Anteil des Zielsignals mehrheitlich tragen, sonst keine Interpretation.

**A7 — In Residual-Netzen ist der mittlere Jacobian identitätsdominiert.** `J_l ≈ I + Korrekturen` (Diagonalmittel ≈ 1.0; Nicht-Identitäts-Anteil 52 % → 16 % mit der Tiefe). Ein Low-Rank-„Workspace" kann auf Transport-Ebene strukturell nicht erscheinen (effektiver Rang 440–503 von 512); die Analyse muss auf den Nicht-Identitäts-Rest bzw. auf gezielte Subraum-Interventionen ausweichen.

**A8 — Interne Anzeigen unter Injektion fester Vektoren enthalten Projektions-Artefakte.** Für einen festen Eingriffsvektor v ist die Verschiebung einer linearen Probe teilweise der konstante Offset α·s·(w·v) — daher das saubere Vorzeichen-Flip zwischen Orth- und Zufalls-Translationen (19c) und Uhr-Shifts von ±0.2–0.4 selbst bei Zufallsrichtungen (20b). Belastbar sind Rotationen (normerhaltend, KL-belegt netzwerk-vermittelt), Selbst-Injektions-Kontrollen (uhr_probe/v_⊥: +18.3, erwartbar) und relative Vergleiche — nie absolute Einzel-Shifts.

**A9 — Rang-1-Zerlegungen haben arbiträre Vorzeichen und input-abhängige Gates.** Der „Schreibpegel entlang einer SPD-Richtung" ist ohne die gelernte Causal-Importance-Funktion fehlspezifiziert: Das Vorzeichen von U_c ist Konvention (V·U fixiert nur das Produkt), und Komponenten sind kontextabhängig aktiviert. Symptom im 21er-Lauf: AUC **unter** Chance (0.304) — wie bei A1 ist Unter-Chance ein Struktur-Warnsignal, kein „keine Information". Statische Lesart-Templates in Auswertungs-Prints verschärfen die Falle; Verdikte müssen konditional aus den Zahlen erzeugt werden.

**A10 — Injizierte Zeit ist kein Befund.** Der Planner bekommt t explizit: `x += time_embedding(t)` wird am Eingang des Encoder-Stacks jeder Position aufaddiert — anders als LLaDA/Dream, wo kein Zeitschritt konditioniert wird. Fortschritts-Dekodierung in der natürlichen Bedingung repliziert damit die Literatur-Referenz („> 0.9 R² aus jeder Schicht"; gemessen 0.97–0.999) exakt — als Durchleitung, nicht als Uhr. Kontrolle: t ≡ 99-Replays der echten Sampling-y_t (das 16er-Design, auf die Schritt-Achse verallgemeinert). *Rest-Vorbehalt: Späte y_t sind unter t ≡ 99 leicht off-distribution; der Zwei-Bedingungs-Vergleich rahmt den Effekt.*

**A11 — Phantom-Parabeln.** PCA glatter, driftender Trajektorien erzeugt *generisch* niederfrequente Polynom-PCs (Random-Walk-PCA → Lissajous-Kurven, „phantom oscillations"). Im Lauf real beziffert: Spektrum-gematchte Surrogate ohne jeden Zeit-Code (Endpunkt-Trend + Phasen-Randomisierung der Residuen pro Dimension) erreichen im Mittel Parabel-R² 0.82–0.94. Ein hoher Parabel-Fit allein ist deshalb **keine** Detektion; belastbar sind die Surrogat-Null (400 Stück, Add-one-p, Bonferroni über Layer), Out-of-Sample-Lesbarkeit aus 2 Dimensionen und Kreuz-Konsistenz über disjunkte Musik-Hälften. *(Nebenlektion aus 23b: Die Histogramm-Baseline wurde zunächst nur geprintet, nicht persistiert — Kennzahlen immer als CSV sichern, vgl. die 21c-Lektion.)*

**A12 — Gesättigte Verhaltensmetriken erzeugen Null-Verdikte.** Die vorregistrierte Steering-Metrik Hold-Rate (Argmax bestätigt das aktuelle y_t) lag im v10-Lauf ab dem zweiten erfassten Schritt konstant bei 1.000 (nur der reine Rausch-Input bei si = 0 liegt bei 0.352) — Folge des dokumentierten Transition-Kollapses (Offener Punkt 5): Der Sampler ist faktisch deterministisch, y_t ist ab si = 1 der Argmax des Vortakts. Das automatische Verdikt meldete „kein Steering" (alle Δ ≡ 0, Spearman = NaN), obwohl KL und Entropie im selben Lauf eine klare, dosis-abhängige Wirkung trugen (in-plane-KL 27× Zufall). Lektionen: (a) vor dem Urteil die Dynamik der Referenzkurve prüfen (Sättigungs-Guard, seit v10.1 in 24c), (b) immer einen kontinuierlichen Zweitkanal mitloggen — ein diskreter Argmax-Kanal kann strukturell blind sein, (c) auffällige Nullen sind Struktur-Warnsignale, kein „kein Befund" (vgl. A1/A9).

---

## 3. Befund 1 — Kein Takt-Integrator (Rhythmus-Zeit-Probes)

Frameweise Targets aus der Beat-Spur (Dim 34), strikte Musik-ID-Splits, drei Baselines (Roh-Frame 35d, ±8-Frame-Kontext 595d, Positions-One-Hot 150d):

| Feature-Basis | Beat (AUC) | Phase (R²) | Frames-bis-Beat (R²) |
|---|---|---|---|
| ±8-Frame-Input-Fenster | 1.000 | **0.683** | 0.599 |
| Roh-Input (1 Frame) | 1.000 | 0.105 | −0.354 |
| Residual Stream L0…L7 | **1.000** | **−0.05 … −0.32** | ≈ 0 |
| MLP-Probe auf L0/L4/L7 | — | **−0.90 … −0.74** | — |

Der Beat selbst ist überall perfekt lesbar (Durchleitung); die Phase — die zeitliche Integration erfordert — ist aus den Aktivierungen weder linear noch per MLP dekodierbar, obwohl ein triviales Kontextfenster sie zu 68 % rekonstruiert. Die MLP-Probes werden sogar stark negativ (Song-Identitäts-Overfit). Verhaltens-Bestätigung: Transition-Grenzen rasten nicht auf Beats ein (Alignment-Quantil q ≈ 0.5 ≈ Zufall; mittlere Distanz 4.18 ≈ Zufallserwartung IBI/4), auch unter Counterfactual-verschobener Beat-Spur.

**Konsequenz:** Ein Finetuning mit Beat-Phasen-Hilfsziel ist jetzt empirisch begründet — es würde eine nachweislich fehlende Fähigkeit adressieren, nicht eine bloß unlesbare.

---

## 4. Befund 2 — Die latente Uhr ist echt (Settledness-Probe + Kontrollen)

Frameweise Probe „Ist dieses Frame schon final?" bei festem Diffusionsschritt (Time-Embedding je Schritt konstant → jedes Signal muss aus dem y_t-Inhalt kommen), Layer-4-Readout, Song-Split:

| Schritt | t | Settled-Anteil | AUC Akt. (alle) | AUC Label (alle) | **AUC Akt. (Mov.)** | **AUC Label (Mov.)** |
|---|---|---|---|---|---|---|
| 30 | 69 | 0.226 | 0.969 | 0.921 | **0.835** | 0.610 |
| 40 | 59 | 0.382 | 0.979 | 0.893 | **0.910** | 0.593 |
| 50 | 49 | 0.553 | 0.981 | 0.884 | **0.894** | 0.585 |
| 60 | 39 | 0.702 | 0.985 | 0.927 | **0.907** | 0.744 |
| 70 | 29 | 0.828 | 0.984 | 0.938 | **0.908** | 0.770 |

Mittlerer Vorsprung über die Label-Baseline: **+0.23 (Movement-only)** vs. nur +0.055 (alle Frames — der Confound A4 dominiert die unkontrollierte Metrik). Mitten in der Trajektorie ist die Label-Baseline fast auf Chance (0.585), die Aktivierungen bei 0.894. Der Residual Stream *weiß* bei festem t und festem Label, welche Movement-Frames stabil sind.

---

## 5. Befund 3 — Nicht in Heads (Attention-Programm-Replacement)

Methode aus „Replacing Attention Heads": symbolische Hypothesen-Programme je Head, Soft-IoU-Bestfit (`sum(min)/sum(max)`), dann Replacement-Experimente (Uniform-Ablation vs. Programm-Ersatz) mit dem Settledness-Gap als Damage-Metrik (Schritt 50, `settled_rate` 0.553).

- **Best-Fit-Landkarte:** Layer 0–4 überwiegend `uniform`/`local_pm8` (Misch-Heads); in Layer 5–7 ein Dutzend Struktur-Heads (`label_boundaries`, `transition_frames`) — die Attention schaut dort gezielt auf die Grenzstruktur der aktuellen Label-Sequenz.
- **Ablations-Ergebnis (Layer 0–4):** Maximaler Einzel-Head-Schaden **−0.014** (L4H4) von +0.31 Gap — kein Head trägt die Uhr. Die Spur ist auf Head-Ebene verteilt/redundant.
- **Methoden-Validierung:** Bei den Heads mit `local_pm8`-Bestfit und messbarem Schaden (L3H3/H4/H7) **rettet das Programm-Replacement den Schaden** (Rescue bis +0.004, L3H7 fast vollständig) — die symbolische Erklärung dieser Heads ist funktional korrekt.
- Layer 5–7: wegen A5 ungetestet (Readout-Position).

---

## 6. Befund 4 — Ein superponiertes Instabilitäts-Ensemble (SPD)

Stochastic Parameter Decomposition (Goodfire AI; `nano_param_decomp`-Referenzimplementierung) auf den zehn MLP-Matrizen der Layer 0–4: je 80 Rang-1-Komponenten `V·U` + Delta-Spillover, CI-Transformer als Causal-Importance-Funktion, Losses Faithfulness / stochastische Rekonstruktion (KL) / Importance-Minimality (p-Anneal). Nach Domänen-Rebalancierung (A6): L0 ≈ 10 aktive Komponenten pro Position, **75,9 % Uhr-Rekonstruktion** (ci 0.854 / zero 0.727 / clean 0.894).

Komponenten-Ablation gegen die Movement-Settled-AUC (799 lebendige Komponenten):

| Konzentrationsmaß | Wert |
|---|---|
| Top-Komponente (L4.linear2 c63) | −0.027 (≈ 21 % des MLP-Uhr-Anteils; 2× bester Head) |
| Top-10-Anteil am Gesamtschaden (0.469) | 20,5 % |
| Top-50-Anteil | 51,3 % |
| **Layer 4** | **55 %** |
| `linear2`-Matrizen (Ausgaberichtungen) | 69 % |

**Urteil:** kein 10-Komponenten-Circuit, aber klar konzentriert — ein **superponiertes Richtungs-Ensemble mit benennbarem Ort** (MLP-Ausgabe von Layer 4), unsichtbar auf Head-/Neuronen-Raster.

**Die Überraschung — invertierte Selektivität:** 14 der 15 schadensstärksten Komponenten feuern *stärker auf unsettled* Movement-Frames (z. B. c21: CI 0.045 settled vs. 0.377 unsettled). Die „Uhr" ist ein **Instabilitäts-Detektor-Ensemble**; Settledness wird als Abwesenheit gelesen. Funktional ist das für einen Denoiser die richtige Größe: nicht „was ist fertig", sondern „wo ist noch Arbeit".

**Nebenbefunde:** Ohne die MLPs der Layer 0–4 hält die Uhr noch AUC 0.727 (Embedding-/Attention-Pfade tragen substanziell); die finalen Logits reagieren auf das komplette Ausschalten dieser MLPs mit nur ~0.004 KL.

---

## 7. Befund 5 — Geometrischer Code vs. kategorische Grammatik (Jacobian-Lens)

Jacobian-Lens (`lens_l(h) = OutputHead(J_l·h)`, `J_l = E[∂h_final/∂h_l]`, Kotangenten-Estimator, bidirektional adaptiert):

- **Spektren:** σ₁/σ₂ = 1.55–1.75, effektiver Rang 440–503/512, `J ≈ I` (A7) — **keine einzelne dominante Transportrichtung**. Uhr-Richtungen (Probe-Vektor, SPD-c63) sind in den Top-Singulärrichtungen schwach, aber überzufällig vertreten (cos bis 0.145 bzw. 0.105; Zufallsniveau ≈ 0.035), wachsend mit der Tiefe.
- **Helle Punkte:** gesättigt (Agreement ≈ 1.0 überall) — an diesen Zuständen kopiert der Planner überwiegend das Input-Label, das schon ab Layer 1 per Output-Head dekodierbar ist. Aussagekräftig erst bei frühen Schritten (offener Punkt).
- **Geometrie-Variation** (Patch des Layer-Outputs; clean: Uhr-Anzeige 0.436, Agreement 1.000):

| Eingriff | Grammatik (Agreement) | Uhr-Verschiebung | Logit-KL |
|---|---|---|---|
| Skalierung ×0.5 … ×2.0 | **1.000 überall** | −0.074 … −0.041 | ≤ 0.003 |
| Rotation Top-2-J, 30° | **1.000** | **−0.144** | 0.002 |
| Rotation Top-2-J, 90° | 0.906 | **−0.523** | 0.528 |
| Orth-Translation β ≤ 1 | 1.000 | −0.13 … −0.20 | ≤ 0.002 |
| Random-Translation β ≤ 1 | ≥ 0.999 | **+0.13 … +0.25** | ≤ 0.005 |

**Dissoziation:** Der Plan-Inhalt (Argmax-Struktur) ist gegen alle moderaten Geometrie-Eingriffe immun — kategorisch-redundant kodiert. Die Uhr-Anzeige kovariiert dagegen systematisch mit Orientierung (Top-J-Ebene) und Amplitude — bei 90°-Rotation kollabiert sie um das ~1,2-fache ihres Clean-Werts, während die Grammatik zu 91 % steht. Die großen KL-Werte der Rotationen belegen einen netzwerk-vermittelten Effekt. *(Vorbehalt: Bei festen Translations-Vektoren ist ein Teil der Verschiebung ein direkter Projektions-Effekt auf die Probe-Richtung — das saubere Vorzeichen-Flip orth vs. random deutet darauf; belastbar sind primär die Rotationen.)*

Die präzisierte Form der Ausgangsthese: nicht *eine* orthogonale Richtung, sondern eine **niedrigdimensionale Orientierungs-Kodierung innerhalb der Top-J-Ebene** trägt die Uhr — der Inhalt liegt außerhalb der Geometrie.

---

## 8. Befund 6 — Unterhalb des Workspace: getragen, still, lesbar (NLA-Analog)

Setup nach dem nla-introspection-Muster: Konzept-Richtungen (6 Genre-Zentroiden, Uhr-Lese-Richtung, SPD-Schreib-Richtung c63, 2 Zufalls-Kontrollen) werden gegen den Top-64-Zeilenraum von J₃ gespalten (`v_J` im „Workspace", `v_⊥` darunter) und bei Layer 3 injiziert. Da der Planner keine Sprache hat, ersetzt ein extern trainierter Aktivierungs-Dekoder (MLP über Layer-6-Aktivierungen, mit `none`-Klasse und Song-Split) den NLA.

**Workspace-Anatomie (20a):** Zufallsniveau = 64/512 = 0.125.

| Konzept | J-Anteil | Lesart |
|---|---|---|
| Genres (6×) | 0.077–0.160 | ≈ Zufall — Genre wird vom Transport nicht bevorzugt |
| Zufalls-Kontrollen | 0.117 / 0.153 | ≈ Zufall — Kalibrierung bestanden |
| **Uhr-Lese-Richtung (Probe)** | **0.050** | 0,4× Zufall — liest bevorzugt *unterhalb* des Workspace |
| **Uhr-Schreib-Richtung (c63)** | **0.448** | 3,6× Zufall — schreibt *in* den Workspace |

**Blindheit des Outputs (20b):** Alle 20 Injektionen (beide Komponenten, alle Konzepte) lassen den Plan unberührt — Agreement ≥ 0.999, Logit-KL 0.002–0.054. Die interne Uhr-Anzeige reagiert dagegen selektiv (Selbst-Injektions-Kontrolle uhr_probe/v_⊥: +18.3; c63: v_⊥ wirkt 12× stärker als v_J); absolute Translations-Shifts stehen unter A8-Vorbehalt.

**Externer Dekoder (20c):**

| | v_⊥ (sub-J) | v_J (Workspace) |
|---|---|---|
| 9 von 10 Konzepten | **1.00** | 0.00 (→ als `none` klassifiziert: nur ~12 % Masse) |
| c63 (einziges Konzept mit J-Anteil 0.45) | 0.56 | 1.00 |

Chance ≈ 0.09 (11 Klassen). **~88 % der Konzeptmasse wird vom Residual-Identitäts-Pfad (A7) bis in die Endschicht getragen, ist am Output kausal still — und extern perfekt dekodierbar.** Die c63-Inversion spiegelt exakt ihre 20a-Zerlegung und dient als eingebaute Positiv-Kontrolle. Konsequenz für Monitoring: Wer diesen Planner nur am Verhalten beobachtet, sieht einen kleinen Ausschnitt seines inneren Zustands; ein einfacher externer Dekoder sieht praktisch alles. *(Einschränkungen: Der Dekoder erkennt injizierte Konzepte, noch nicht natürlich vorkommende Sub-J-Inhalte; Blindheit gilt für Dosis α = 1.0. Die False-Positive-Rate ist inzwischen dokumentiert: **0.000**.)*

---

## 9. Befund 7 — Geheimnis-Extraktion am Tatort: strukturiert, kausal, aber gegated

Konzept nach mlp-secret-extraction: Aus den L4-MLP-Gewichten allein werden per Input-Optimierung (Random-Restarts, Neuron-Seeds aus den W₁-Zeilen, komponierte Paare) die bevorzugten Eingaben der Uhr-Schreib-Richtungen (`g(z) = u·MLP₄(z)`, Top-5-SPD-Ensemble + c63) rekonstruiert.

**Trigger-Landschaft (21b):** ~120 Starts kollabieren in **4 Basins** (Dedup |cos| > 0.9) — hochstrukturiert, aber mit Halo-Abfällen von nur ~0.7 % bei 10 %-Störungen: **breite Plateaus, keine scharfen Lookup-Schlüssel**. Das beste Optimum stammt aus einem Neuron-Seed, Ränge 1–3 aus Kompositionen — die Repo-These „die Gewichtszeilen enthalten die Suchrichtungen" repliziert. Alle vier Optima liegen im Embedding-Umfeld von Movement-Label 67 (dann 0/86/51/34), und ihre Schreib-Vektoren Lens-dekodieren ebenfalls zuerst auf 67/86/34 — die Uhr-Schreiber sind um das **dominante Bewegungs-Vokabular** des Planners organisiert (vgl. `top_moves` [68, 67, …] aus Abschnitt 14); die Kosinusse zu den settled/unsettled-Zentroiden bleiben klein (±0.1): Instabilität wird offenbar als *Label-Flattern zwischen Favoriten* detektiert, nicht als ungerichtetes Rauschen.

**Validierung gegen das Settledness-Maß (21c):** Der signierte Schreibpegel als Instabilitäts-Detektor auf echten Frames: **AUC 0.304 (Ensemble) / 0.242 (c63)** — unter Chance; vorzeichen-korrigiert 0.696/0.758 — über der Label-Baseline (0.586), aber weit unter der Probe (0.894). Kausal ist die Top-Trigger-Richtung dennoch potent: Injektion verschiebt die Uhr-Anzeige um **−1.93** gegenüber Zufall +0.33 ± 0.81 (≈ 2,8σ; A8-Vorbehalt für absolute Werte).

**Urteil:** Secret-Extraction im strengen Sinn **nicht bestanden** — und genau das ist der Befund: Die Uhr liegt nicht in festen, vorzeichen-stabilen Richtungen, sondern im **CI-gegateten Zusammenspiel** input-abhängig aktivierter Komponenten (A9). Das schließt konsistent an Befund 4 (Superposition) und Befund 5 (Geometrie-Code) an: ohne das Gate keine Gewichts-Rekonstruktion. Der Weg dahin — Weight-Reader über die Trainings-Population (Phase 2) bzw. gate-bewusste Objectives — ist damit präzise motiviert.

---

## 10. Befund 8 — Die Parabel ist echt, die lesbare Schritt-Uhr ist Oberfläche (Trajektorien-Geometrie)

Test der Latent-Clock-Befunde aus Diffusions-LMs (τ_dlm-Vorlage; LLaDA/Dream: Fortschritt aus jeder Schicht > 0.9 R² dekodierbar; die Zeit-Repräsentation bildet eine geordnete, niedrigdimensionale **parabolische Kurve** im Aktivierungsraum) auf dem Planner: 24 echte Sampling-Trajektorien, per-Schritt/-Layer-Mittelaktivierungen (50 Stützstellen × 8 Layer), Musik-ID-Splits — in zwei Bedingungen: **natürlich** (echtes t) und **t-fest** (t ≡ 99, Replays der echten y_t; A10).

**Lesbarkeit (23b):** Natürlich R² 0.970–0.999 in jeder Schicht — die Literatur-Referenz repliziert exakt, ist hier aber reine Durchleitung des injizierten Zeit-Embeddings (A10). t-fest wächst R² mit der Tiefe von −0.09 (L0) auf 0.68 (L7, ab L2 monoton) und bleibt damit **unter der Label-Histogramm-Baseline** von y_t — Verdikt des Laufs: „der lesbare Fortschritt ist Oberflächenstatistik von y_t". Bemerkenswert: Die reine Top-2-PC-Projektion (quadratische Features) liest mit 0.75–0.80 in *jeder* Schicht besser als die volle 512d-Ridge — die Trajektorien-Geometrie generalisiert über Musik-Gruppen, wo die hochdimensionale Probe an Song-Richtungen overfittet.

**Geometrie (23c/23d), t-fest:**

| Layer | R²_512d | R²_Top2 | EV Top-2 | R²_Parabel | Kreuz-Konsistenz | p (vs. Surrogate) |
|---|---|---|---|---|---|---|
| 0 | −0.09 | 0.80 | 0.9987 | 0.892 | −57.2 | 0.995 |
| 1 | 0.21 | 0.78 | 0.9986 | 0.984 | −2.11 | **0.0025** |
| 2 | 0.08 | 0.78 | 0.9991 | 0.993 | 0.75 | **0.0025** |
| 3 | 0.26 | 0.77 | 0.9995 | 0.951 | 0.53 | 0.095 |
| 4 | 0.41 | 0.77 | 0.9997 | 0.944 | 0.66 | 0.187 |
| 5 | 0.59 | 0.76 | 0.9997 | 0.954 | 0.90 | 0.052 |
| 6 | 0.64 | 0.75 | 0.9997 | 0.977 | **0.91** | **0.0025** |
| 7 | 0.68 | 0.76 | 0.9994 | 0.973 | 0.77 | **0.0025** |

Die Schritt-Mittel-Trajektorie ist extrem niedrigdimensional (EV Top-2 ≥ 0.9986) und perfekt geordnet (|Spearman(PC1, τ)| ≥ 0.9997); der Endpunkt-Trend trägt ~95 % ihrer Varianz, die Parabel lebt im kleinen Residuum. **Der A11-Ernstfall trat ein:** Die spektrum-gematchten Surrogate (400 Stück, Add-one-p, Bonferroni 0.05/8 = 0.0063) erreichen im Mittel selbst Parabel-R² 0.92–0.94 — ein naiver Parabel-Fit hätte in allen 16 Bedingungs-Layer-Kombinationen „Parabel!" gemeldet. Real übersteht die Detektion die Null in den Layern 1–2 und 6–7 (p = 0.0025, das Minimum bei 400 Surrogaten); die Kreuz-Konsistenz über Musik-Hälften ist nur in der Tiefe solide (0.90/0.91 bei L5/L6; in L0–L1 ist die Rest-Krümmung hälften-idiosynkratisch, xval bis −57). **In Layer 6 kommt alles zusammen:** R²_Parabel 0.977, p 0.0025, Kreuz-Konsistenz 0.91.

**Natürliche Bedingung als Kontrast:** Das injizierte Sinusoid-Embedding krümmt die Trajektorie massiv (Endpunkt-Trend-Anteil −3 bis −9, schlechter als der Mittelwert), und in frühen Layern ist die Kurve *weniger* parabolisch als ihre Surrogate (p 0.75–1.0). Erst mit der Tiefe formt das Netz daraus eine saubere Parabel (L4–L7: R² 0.986–0.992, p 0.0025, Kreuz-Konsistenz bis 0.98) — der Planner transformiert den injizierten Zeit-Code mit der Tiefe in dieselbe Geometrie-Klasse, die auch der Inhalts-Pfad zeigt.

**Urteil:** Von den drei Latent-Clock-Thesen der Vorlage überlebt (1) „> 0.9 R² aus jeder Schicht" nur als A10-Artefakt, (2) „geordnete, niedrigdimensionale Struktur" vollständig, (3) „parabolische Kurve" als echte, aber **tiefen-abhängige und null-geprüfte** Signatur — am klarsten in Layer 6. *(Nachtrag v10.1: Die Histogramm-Baseline liegt bei **r2_hist = 0.890** (`planner_parabola_baseline.csv`, seit v9.1 persistiert) — damit bleibt auch die beste Lesart (Top-2, 0.80) klar unter der Oberflächen-Statistik von y_t; Offener Punkt 12 ist geschlossen.)*

**Deutung (LifeNNgine-Linse) — Hypothese, noch ungetestet:** [`Erikiss/LifeNNgine`](https://github.com/Erikiss/LifeNNgine) zeigt am Game of Life, dass **band-selektive Automaten-Regeln** (überlebe genau bei Nachbarsumme 2–3 — eine intrinsisch nicht-monotone, parabelförmige Funktion) mit monotonen Aktivierungen schwer, mit **parabolischen Aktivierungen** (lernbares Grad-2-Polynom „PolyKAN", `Square` x², `Gaussian`) dagegen dramatisch leichter lernbar sind — minimale L(1,1)-Netze genügen. Der Planner-Befundkomplex trägt exakt die Signatur einer band-selektiven Berechnung, die von monotonen GELU-Einheiten nur *simuliert* wird: die Uhr ist ein Abweichungs-/Instabilitäts-Detektor (Befund 4, invertierte Selektivität; Befund 7: Label-Flattern zwischen Favoriten), sie ist **superponiert statt kompakt** (Befund 4: Rang-1-Ensemble, kein sparsames Subnetz), **gegated statt gewichts-lesbar** (Befund 7: ohne CI-Gate keine Extraction) — und ihre Prozess-Geometrie ist eine **Parabel** (Befund 8). Die LifeNNgine-Lesart: Superposition und Gating sind hier möglicherweise kein Design, sondern die *Kosten* der falschen Aktivierungs-Basis. Testbare Vorhersage für Phase 2 (Achse G, `--activation poly2/square/gaussian` seit v10 im Trainings-Code): Mit parabolischer FF-Aktivierung sollte dieselbe Uhr (i) **kompakter** werden (SPD-Konzentration ↑, weniger Komponenten tragen mehr Schaden), (ii) **gewichts-lesbar** (die 21er-Secret-Extraction besteht ohne Gate), (iii) die Parabel-Geometrie **schärfer/früher** (23er-Assay), bei (iv) mindestens gleicher Verhaltens-Gesundheit (BAS, `transition_frac`).

---

## 11. Befund 9 — Steuerbar ist die Ebene, nicht der Fahrplan (Parabel-Steering)

Kausaltest der vierten Latent-Clock-These („die innere Uhr lässt sich steuern") auf der natürlichen L6-Kurve aus Befund 8: Injektion entlang der lokalen Kurven-Tangente (Dosis α in lokalen Bogen-Schritten, ±0.5–4) in echte Sampling-Zustände an vier Eval-Punkten, gemessen am Verhalten (A5/A8-konform); Kontrollen: Kurven-Normale (in der PC1-2-Ebene, tangential-orthogonalisiert) und drei Zufallsrichtungen gleicher Norm.

**Der A12-Zwischenfall:** Die vorregistrierte Hold-Rate war ab dem zweiten erfassten Schritt (si = 1) auf 1.000 gesättigt (Transition-Kollaps) — das v10-Verdikt lief blind auf Null. Der Signal-Kanal ist die **Logit-Entropie**, deren Referenzkurve sauber monoton von 4.61 auf 0.012 fällt (v10.1: Sättigungs-Guard + sensitivitäts-gewichtetes Verdikt).

**Ergebnis auf dem Entropie-Kanal** (ΔS = Entropie-Verschiebung; effektive Schritte = ΔS / lokale Referenz-Steigung):

| Eval-Punkt | S_ref | Spearman(α, ΔS) | eff. Schritte bei α = −4 / −2 / +2 / +4 |
|---|---|---|---|
| si = 10 (τ ≈ 0.20) | 0.635 | **−1.00** | −3.8 / −1.7 / +1.4 / +2.4 |
| si = 20 (τ ≈ 0.40) | 0.261 | **−1.00** | −2.6 / −1.1 / +0.6 / +0.7 |
| si = 30 (τ ≈ 0.61) | 0.111 | −0.43 | −1.4 / −0.5 / −0.0 / −0.6 |
| si = 40 (τ ≈ 0.81) | 0.039 | +0.57 | −0.1 / +0.0 / −0.1 / −0.4 |

- **Früh und mittig ist das Steering perfekt:** dosis-monoton (Spearman −1.00), vorzeichen-richtig (vorwärts → schärfer, rückwärts → flacher; Asymmetrie 75 %), sensitivitäts-gewichtet ρ = **−0.87**. Rückwärts-Steering übersetzt fast 1:1 (α = −4 → −3.8 Schritte), vorwärts sublinear (Kompression Richtung Entropie-Boden). Spät (si ≥ 30) ist nichts mehr zu steuern — die Referenz liegt am Boden.
- **Die Wirkung ist kurven-verstärkt:** |ΔS| der Tangente bei |α| = 2 ist 12× Zufall (0.031 vs. 0.0025 ± 0.0027, z = 10.3); die in-plane-KL ist 27× der Zufalls-KL — das Netz verstärkt genau die Kurvenebene, Zufallsrichtungen verhallen.
- **Aber: die Normale wirkt 1.5× stärker als die Tangente** (0.046 vs. 0.031) — die Wirkung ist **ebenen-, nicht tangenten-spezifisch**. Zusammen mit Befund 5 (Uhr kovariiert mit Orientierung in der Top-J-Ebene) und Befund 8 (EV Top-2 ≈ 1): Die 2D-Kurvenebene ist der kausale Uhr-Kanal des Planners; die exakte Tangentenrichtung ist kein privilegierter Fahrplan-Regler.
- **Die Inhalt/Prozess-Dissoziation repliziert unter Steering:** agree_clean = 1.000 überall — bis |α| = 4 kippt kein einziger Argmax. Verschieben in der Uhr-Ebene ändert Konfidenz/Schärfe, nie den Plan.

**Verdikt (v10.1, konditional):** gerichtet ✓ (ρ_w −0.87, z 10.3, Asymmetrie 75 %), kurven-spezifisch ✗ → **These 4 hält in der mittleren Lesart.** *(Vorbehalt: Die natürliche L6-Ebene enthält auch den Durchleitungs-Anteil des injizierten Zeit-Embeddings (A10); welcher Teil der Ebenen-Wirkung Embedding-Spoofing und welcher Inhalts-Uhr ist, klärt erst die Zerlegung gegen die t-fest-Ebene — Offener Punkt 15.)*

---

## 12. Synthese

> Der AtomicDance-Planner hört keinen Takt, aber er beobachtet sich selbst: Er unterhält eine frameweise, geometrisch kodierte Instabilitäts-Landkarte seines eigenen Denoising-Prozesses — implementiert als superponiertes Ensemble von Rang-1-Richtungen in der MLP-Ausgabe von Layer 4, **geschrieben in den transportierten Workspace, gelesen unterhalb davon** — während der Plan-Inhalt davon getrennt, kategorisch und eingriffsrobust repräsentiert ist. Der Großteil dessen, was im Stream liegt, erreicht den Output nie — bleibt aber von außen vollständig lesbar.

Einordnung: Die Spur ist das frameweise D3PM-Analogon eines Verifikations-Gates (vgl. Termination-Circuits in Reasoning-LMs); die Inhalt/Prozess-Trennung und die Sub-Workspace-Lesbarkeit spiegeln Workspace- und NLA-Befunde aus LLMs im Miniaturformat — mit der residual-bedingten Einschränkung A7, dass „Transport-Subräume" hier nicht low-rank sind. Die Parabel-Detektion (Befund 8) schärft diese Trennung auf der Schritt-Achse: Die Prozess-*Geometrie* (geordnete, niedrigdimensionale, in der Tiefe parabolische Trajektorie) ist real und null-geprüft, während die *lesbare* Schritt-Uhr nicht über die Oberflächenstatistik von y_t hinauskommt — der Planner braucht keine tiefe Schrittzählung, weil die Restarbeit frameweise kodiert ist (Befund 4). Befund 9 macht die Geometrie kausal: Verschieben in der Kurvenebene verschiebt die Verhaltens-Schärfe dosis-treu — und nur dort, Zufallsrichtungen verhallen 27-fach gedämpft —, während der Plan-Inhalt keinen einzigen Frame nachgibt. Acht Methoden, ein konsistentes Bild; jede Methode hat mindestens ein Artefakt der vorherigen korrigiert — und der einzige „Fehlschlag" (Befund 7) ist selbst ein Strukturbefund: Die Uhr ist gegated, nicht statisch.

---

## 13. Offene Punkte

1. **Layer-5–7-Struktur-Heads** gegen die Uhr testen (Probe auf Layer 7 oder Verhaltens-Damage) — A5 auflösen.
2. **Top-k-Gemeinschafts-Ablation** der SPD-Komponenten (Einzelablation unterschätzt Redundanz) + Schreibrichtungs-Check (cos(U_c63, Probe-Richtung)).
3. **Helle Punkte bei frühen Schritten** (t ≈ 90–99), wo der Planner noch nicht kopiert.
4. **Geometrie-Variation replizieren** mit Patch-Layer ≠ Probe-Layer und gemittelten Zufallsvektoren.
5. **Sampler-Kollaps prüfen:** `transition_frac` der Samples (0.77–1.0) gegen die Ground-Truth-Rate (`(test_labels == 0).mean()`) — möglicher eigenständiger Trainingsbefund. *Der v10-Lauf liefert Verhaltens-Evidenz: Der Argmax bestätigt y_t ab dem zweiten Schritt zu 100 % (A12) — der Sampler ist faktisch deterministisch.*
6. **Genre-Frage** ist mit diesem Test-Split unbeantwortbar (A1) — Probing-Fenster aus dem Train-Split ziehen (mehrere Songs pro Genre), Planner bleibt eingefroren.
7. **Finetuning mit Beat-Phasen-Hilfsziel** — jetzt empirisch begründet (Befund 1).
8. **Natürliche Sub-J-Inhalte lesen** (Befund 6 nutzt injizierte Konzepte): z. B. die Uhr rein aus Sub-J-Projektionen dekodieren; dazu Dosis-Wirkungs-Kurve der Output-Blindheit und Dokumentation der False-Positive-Rate.
9. **Gate-bewusste Secret Extraction (Folge aus Befund 7):** vorzeichen-getrennte Objectives (±u), CI-gegatete Zielfunktion (mit `ci_fn18`), und eine SPD-Basis-Probe (Features `(z·V_c)` der Top-Komponenten → winzige logistische Probe) als Zwischenstufe „Gewichte + minimale Kalibrierung"; der Weight-Reader über die Trainings-Population bleibt der Phase-2-Anschluss.
10. **Phase 2 — Trainingskontext-Variation:** Literatur- und Versuchsplan liegen in [`training_context_literature.md`](training_context_literature.md) (Achsen Seeds/Daten/Schedule/Beat-Loss/`--transition-weight`/Architektur × v6-Assay).
11. **Parabel-Steering (vierte These der Latent-Clock-Vorlage):** Injektion entlang der lokalen Parabel-Tangente/-Normale in Layer 6. — **Erledigt in Befund 9:** gerichtet und dosis-treu, aber ebenen- statt tangenten-spezifisch (mittlere Lesart).
12. **Histogramm-Baseline nachtragen:** — **Erledigt (v10.1):** r2_hist = 0.890 (13/11 Musik-Gruppen Train/Test); auch die Top-2-Lesart (max 0.80) bleibt unter der Baseline — Befund 8 entsprechend geschärft.
13. **Abschnitt 22 (Spektrale Volumetrie) in den Report übernehmen:** Der v9-Lauf hat 22a/22b mit ausgeführt (k*-Volumen, PR-Karten, `planner_jspace_volume.csv` / `planner_spectral_assay*.csv`) — Zahlen sichten und als Befund nachtragen.
14. **Achse G ausführen (LifeNNgine-Hypothese aus Befund 8):** Planner-Retrain mit `--activation poly2` (lernbare Parabel; Kontrollen: `square`, `gaussian`, GELU-Baseline, je ≥ 5 Seeds) und den v6/v9-Assays prüfen, ob die Uhr kompakt, gewichts-lesbar und geometrisch schärfer wird — Code seit v10 in `train_atomic.py`/`atomic_planner.py`, Checkpoint-Rekonstruktion in Notebook-Zelle 12 liest `activation` bereits aus den Args.
15. **Ebenen-Anatomie des Steerings (Folge aus Befund 9):** Die natürliche L6-Ebene enthält den Durchleitungs-Anteil des injizierten Zeit-Embeddings (A10). Zerlegung der in-plane-Wirkung: Steering-Chart aus der t-fest-Kurve wiederholen und beide gegen die explizite `time_embedding(t) − time_embedding(t′)`-Differenzrichtung vergleichen — wie viel der Ebenen-Wirkung ist Embedding-Spoofing, wie viel Inhalts-Uhr?

---

## Anhang: Werkzeuge, Artefakte, Versionen

**Werkzeuge (Forks):** [`Erikiss/explaining_attention_heads`](https://github.com/Erikiss/explaining_attention_heads) (Programm-Replacement) · [`Erikiss/param-decomp`](https://github.com/Erikiss/param-decomp) (`nano_param_decomp`-SPD) · [`Erikiss/jacobian-lens`](https://github.com/Erikiss/jacobian-lens) (J-Lens) · [`Erikiss/nla-introspection`](https://github.com/Erikiss/nla-introspection) (Sub-Workspace-Dekodierung) · [`Erikiss/mlp-secret-extraction`](https://github.com/Erikiss/mlp-secret-extraction) (Abschnitt 21). Beispiel-Vorlagen: subliminal-coax-Notebooks 01–04 (Beat-Tabelle, τ_dlm-Latent-Clock, Puppet-Time-Probes, CoAx).

**Ergebnis-Artefakte** (`MyDrive/AtomicDance/mechinterp/outputs/`): `planner_linear_probes.csv` · `planner_rhythm_frame_probes.csv` · `planner_phase_linear_vs_mlp.csv` · `planner_beat_alignment.csv` · `planner_settledness_probe.csv` · `planner_settledness_controls.csv` · `planner_head_program_bestfit.csv` · `planner_head_replacement_gap.csv` · `planner_spd_component_attribution.csv` · `planner_jacobian_lens.pt` · `planner_jspace_spectra.csv` · `planner_geometry_variation.csv` · `planner_concept_jsplit.csv` · `planner_subj_blindness.csv` · `planner_subj_decoder.csv` · `planner_mlp4_secrets.csv` · `planner_mlp4_secret_validation.csv` · `planner_jspace_volume.csv` · `planner_spectral_assay.csv` · `planner_spectral_assay_clock.csv` · `planner_parabola_decode.csv` · `planner_parabola_geometry.csv` · `planner_parabola_null.csv` · `planner_parabola_baseline.csv` · `planner_parabola_steering.csv` · `planner_parabola_steering_ref.csv` · `planner_parabola_steering_verdict.csv` (+ zugehörige PNGs).

**Notebook-Versionen:** v2 Modell-Rekonstruktion verifiziert (`d5bfc58`) → v2.1 sBM-Guard (`750204e`) → v3 Rhythmus-/Settledness-Probes (`0c55641`) → v3.1 Entscheidungszellen (`97b63e0`) → v3.2 Head-Replacement (`b32ff0e`) → v4/v4.1 SPD (`c628d4e`, `1fbf639`) → v5 Jacobian-Lens (`56cdebd`, `dba1f6c`) → v6 Tiefenbohrung (`2c70181`, `1ecb214`, `db784b6`) → v7 Secret Extraction (`7d1ef11`, `f5bc905`) → v8 Spektrale Volumetrie (`745a711`) → v9 Parabel-Detektion (`e90200d`, `016e2ca`, `88a47ba`) → v9.1 Baseline-Persistierung (`8aad87e`) → v10 Parabel-Steering + Achse G (`0eae642`, `629860a`, `9ef9b41`) → v10.1 Verdikt-Kanal-Guard + Befund 9.
