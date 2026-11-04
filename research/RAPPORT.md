# Rapport de recherche : modèle quantitatif multi-cryptos

*Novembre 2026 · simulation uniquement · aucun argent réel*

## Verdict

**Le modèle n'est pas déployé.** La meilleure variante (V3, tendance + régime BTC) a échoué à la règle
de décision fixée avant le test final : sur l'année jamais vue (nov. 2025 → nov. 2026), elle perd
**−9,3 % après frais**. Elle protège mieux que le bitcoin (pire baisse 23 % contre 47 %) mais ne gagne
pas d'argent. La stratégie actuelle du bot (EMA, bougies 4 h) a fait mieux sur cette même année
(−2,3 %, pire baisse 9 %) et reste en place.

## Protocole

- **Données** : 14 cryptos en EUR (Bitstamp, API publique), journalier, jusqu'à 10 ans d'historique
  (`research/data`, collecte par `research/fetch_data.py` via GitHub Actions).
- **Séparation** : développement 2019 → oct. 2025 ; **test final unique** nov. 2025 → nov. 2026,
  jamais utilisé pour choisir quoi que ce soit.
- **Coûts** : 0,45 % par euro échangé (frais taker Kraken 0,40 % + glissement 0,05 %).
- **Modèle ML (V0)** : 14 facteurs (momentum multi-horizons, momentum relatif, retournement, tendance,
  régime de volatilité, choc de volume, distance au plus haut, tendance du BTC), ensemble ridge +
  gradient boosting, entraînement walk-forward trimestriel avec purge et embargo, volatilité cible 30 %.
- **Variantes déclarées avant d'être testées** : V1 (ML lissé, hebdo), V2 (tendance multi-horizons
  sans ML), V3 (V2 + filtre de régime BTC). Code : `research/quant_model.py`, `research/variants.py`.

## Résultats

| Stratégie | Dév. 2019-2025 : CAGR / Sharpe / pire baisse | Test 2025-2026 : rendement / pire baisse |
|---|---|---|
| V0 ML quotidien | −1,5 % / 0,12 / 64 % | −53,3 % / 60 % |
| V1 ML lissé hebdo | +24,4 % / 0,83 / 40 % | −17,4 % / 39 % |
| V2 Tendance multi-horizons | +22,6 % / 0,81 / 48 % | −24,2 % / 31 % |
| **V3 Tendance + régime BTC** | +24,3 % / 0,85 / 43 % | **−9,3 % / 23 %** |
| Buy & hold BTC | +48,6 % / 0,95 / 74 % | −25,1 % / 47 % |
| Bot actuel (EMA 4 h, BTC/ETH/SOL) | +15,8 % / 0,77 / 23 % | −2,3 % / 9 % |

## Ce qu'on apprend

1. **Le modèle ML n'a aucun pouvoir prédictif.** Corrélation moyenne prévision/réalité : 0,004
   (t = 0,46). Sa rotation (61× le capital par an) coûtait 28 % par an en frais.
2. **Le suivi de tendance réduit le risque mais n'apporte pas de rendement en plus.** Sur 6 ans, aucune
   variante ne bat le Sharpe du simple achat de bitcoin ; leur apport est de diviser la pire baisse
   par ~1,7.
3. **La simplicité a gagné hors échantillon.** Le bot actuel, le plus simple et le plus prudent, a le
   mieux résisté sur l'année de test.

## Limites relevées par l'audit indépendant

Aucun biais d'anticipation ni fuite d'information n'a été trouvé, et la comptabilité des frais est
correcte. Les points suivants font tous paraître les variantes **meilleures** qu'elles ne sont, ce qui
renforce la décision de ne pas déployer :

- **Biais du survivant** : l'univers = 14 cryptos encore cotées aujourd'hui (les disparues manquent).
- **Plafond de 35 % par actif non garanti** après la mise à l'échelle de la volatilité cible.
- **Correction pour tests multiples optimiste** : seuls 4 essais comptés, alors que des choix de
  paramètres ont précédé.
- La référence « équipondéré mensuel » était en réalité rééquilibrée chaque jour (pénalisée d'environ
  2,5 points par an), et le Sortino est calculé de façon non standard (surestimé).
- Prix manquants traités comme un rendement nul (effet faible, de signe variable).

## Pistes si on poursuit

Corriger les points ci-dessus, puis tester des sources de rendement différentes du prix seul
(taux de financement des contrats perpétuels, flux on-chain), qui sont celles qu'utilisent réellement
les fonds crypto. Toute nouvelle idée devra être jugée sur une **nouvelle** période jamais vue :
l'année de test actuelle a été consommée.

---

# Volet actions (ETF) : novembre 2026

**Verdict : aucune stratégie active déployée.** Trois stratégies classiques déclarées à l'avance
(`research/stocks.py`), en euros, frais Interactive Brokers inclus (0,10 % par euro échangé).
Sélection sur 2004-2019, test unique sur 2020 → nov. 2026.

| Stratégie | 2004-2019 : CAGR / Sharpe / pire baisse | 2020-2026 : CAGR / Sharpe / pire baisse |
|---|---|---|
| S1 Allocation tactique (tendance) | +7,1 % / 0,57 / 16 % | +5,4 % / 0,57 / 14 % |
| S2 Double momentum | +9,8 % / 0,59 / 22 % | +11,7 % / 0,69 / 27 % |
| **S3 Rotation sectorielle filtrée** (retenue) | +12,0 % / 0,72 / 21 % | **+5,7 % / 0,42 / 23 %** |
| S&P 500 détenu | +9,8 % / 0,57 / 51 % | +15,0 % / 0,77 / 33 % |
| 60/40 actions/obligations | +8,4 % / 0,63 / 26 % | +8,9 % / 0,68 / 19 % |

S3, la meilleure en développement, échoue au test (Sharpe 0,42 contre 0,77 pour le S&P 500 détenu).
Comme pour la crypto : les règles actives réduisent les pires baisses mais ne battent pas l'indice
détenu une fois hors échantillon. Univers d'ETF larges : pas de biais du survivant ; données Yahoo
Finance (cours ajustés des dividendes), risque de change EUR/USD inclus.
