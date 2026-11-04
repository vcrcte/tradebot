# Bot de trading crypto (utilisable depuis la France)

> **Dépôt privé.** Ce dépôt fait tourner le bot **en simulation** (argent fictif) toutes les heures grâce à
> GitHub Actions (`.github/workflows/bot.yml`). Chaque passage enregistre `state_paper.json` et régénère
> `docs/index.html` (le tableau de bord, données intégrées). L'app « Bot de trading » sur le Mac récupère
> ce dépôt toutes les 10 minutes et affiche le tableau de bord en local.
> Les achats, ventes et erreurs sont publiés en commentaire de l'issue « Alertes du bot de trading ».
> Le workflow refuse de tourner si `config.yaml` n'est pas en `mode: paper` : aucune clé API n'est stockée ici.

Bot Python qui trade du BTC, de l'ETH ou du SOL contre l'euro sur **Kraken**, un exchange agréé MiCA (via Payward Europe) et donc autorisé en France. Il démarre en **simulation** : il suit les vrais prix, mais n'envoie aucun ordre.

## Ce qu'il fait

À chaque clôture de bougie (4 h par défaut), pour chaque paire :

1. Il récupère les prix et calcule les indicateurs : EMA 20/50/200, RSI, ATR.
2. **Entrée** : l'EMA 20 est au-dessus de l'EMA 50, le prix au-dessus de l'EMA 200 (tendance de fond haussière) et le RSI sous 70, ces conditions étant apparues depuis au plus 3 bougies (`entry_window`).
3. **Taille de position** : le risque par trade suit un demi-Kelly calculé sur l'historique des trades (1 % tant qu'il y a moins de 20 trades), plafonné à 2 %. Une position ne dépasse jamais 30 % du capital.
4. **Sortie** : un stop suiveur à 3 × ATR, ou l'EMA 20 repassée sous l'EMA 50.
5. **Coupe-circuits** : à -5 % sur la journée, le bot n'ouvre plus rien jusqu'au lendemain. À -25 % depuis le plus haut, il s'arrête complètement.
6. Il envoie une alerte Telegram à chaque achat, vente ou arrêt.

Il n'achète que ce qu'il paie comptant : pas de levier, pas de vente à découvert.

## Installation (environ 10 minutes)

```bash
cd tradebot
python3 -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # remplis Telegram si tu veux les alertes
```

## Étape 1 : backtest

```bash
python backtest.py                      # ~30 jours de bougies 1 h (limite de l'API Kraken)
python backtest.py --csv XBTEUR_60.csv  # historique long : fichiers OHLCVT téléchargeables
                                        # sur le site de support Kraken ("Downloadable historical OHLCVT")
```

Le backtest utilise exactement le même moteur que le bot, avec 0,40 % de frais par ordre et du slippage. Il affiche le rendement, la comparaison avec un simple achat-conservation, le drawdown max, le ratio de Sharpe et le taux de trades gagnants, et enregistre la courbe de capital dans `backtest_equity.png`.

**Si la stratégie ne bat pas le buy & hold après frais sur 2 ou 3 ans de données, ne la passe pas en réel.** C'est le cas le plus fréquent.

## Étape 2 : simulation sur les prix en direct (au moins 4 semaines)

```bash
python bot.py           # un passage
python bot.py --loop    # en continu, vérifie toutes les 5 min
python bot.py --status  # capital, positions, statistiques
```

Pour le lancer automatiquement toutes les 15 minutes :

- **Mac/Linux** : `crontab -e`, puis ajoute
  `*/15 * * * * cd /chemin/vers/tradebot && .venv/bin/python bot.py >> bot.log 2>&1`
- **Windows** : utilise le Planificateur de tâches avec `.venv\Scripts\python.exe bot.py`.

L'ordinateur doit rester allumé. Pour que le bot tourne ordinateur fermé, il faut un petit serveur (un VPS à quelques euros par mois) avec les mêmes commandes.

## Étape 3 : passage en réel (seulement si l'étape 2 est concluante)

1. Sur Kraken : *Settings → API → Create key*, avec les droits **Query funds** et **Create & modify orders** uniquement. **Ne coche jamais « Withdraw ».**
2. Ajoute la clé et le secret dans `.env`.
3. Dans `config.yaml` : `mode: live` et `capital_cap` = la somme maximale que le bot peut utiliser, par exemple 200 €.
4. Lance `python bot.py --live`. Il faut les deux réglages à la fois, ce qui évite de passer en réel par erreur.

Le bot ne touche jamais plus que `capital_cap`, et ce budget baisse avec ses pertes. La simulation et le réel ont des fichiers d'état séparés (`state_paper.json`, `state_live.json`).

## À savoir

- **Limite** : en réel, les stops sont vérifiés à chaque passage du bot. Ce ne sont pas des ordres stop posés sur l'exchange, donc en cas de chute brutale entre deux passages, la sortie se fait plus bas que le stop prévu.
- **Fiscalité** : en France, les plus-values crypto sont imposables quand tu convertis en euros (les échanges crypto contre crypto ne le sont pas). Un compte sur un exchange étranger se déclare chaque année (formulaire 3916-bis). Vérifie les règles en vigueur sur impots.gouv.fr.
- **Avertissement** : ce code est un outil, pas un conseil en investissement. Aucune stratégie ne garantit un gain, et la plupart des stratégies simples perdent après frais. N'y mets que ce que tu acceptes de perdre entièrement.

## Fichiers

| Fichier | Rôle |
|---|---|
| `config.yaml` | Tous les réglages : paires, timeframe, risque, mode |
| `bot.py` | Exécution (simulation ou réel), planification, statut |
| `backtest.py` | Test sur l'historique avec le même moteur |
| `engine.py` | Logique de trading, taille de position, coupe-circuits |
| `strategy.py` | Indicateurs et signaux (le fichier à modifier pour changer de stratégie) |
| `broker.py` | Ordres simulés ou réels (via ccxt) |
| `notifier.py` | Alertes Telegram |
