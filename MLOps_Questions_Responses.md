# Réponses aux Questions d'Architecture & Choix Techniques MLOps

Ce document détaille les choix techniques, l'architecture et les processus mis en place ou recommandés pour le projet MLOps actuel.

## 1. MLOps - Gestion du Model Registry

### Stratégie de Versioning et Rollbacks
*   **Versioning** : Nous utilisons le versioning natif de MLflow. Chaque entraînement (`run`) qui produit un modèle satisfaisant est enregistré dans le Model Registry. MLflow incrémente automatiquement le numéro de version (v1, v2, v3...).
*   **Cycles**: Nous utilisons les "Stages" (phases) de MLflow : `None`, `Staging`, `Production`, `Archived`.
*   **Rollbacks** : En cas de régression en production :
    1.  Identification de la version précédente stable (ex: v1).
    2.  Transition de la version actuelle (v2) de `Production` vers `Archived` (ou `Staging` pour analyse).
    3.  Transition de la version stable (v1) vers `Production`.
    4.  L'API, configurée pour charger dynamiquement le modèle `Production` (ou au redémarrage), repassera sur l'ancienne version.

### Hébergement et Haute Disponibilité
*   **Actuel (POC)** : Hébergement local via Docker (`sqlite` pour le backend store, système de fichiers pour les artefacts).
*   **Cible Production** : Hébergement Cloud (ex: AWS/Azure/GCP).
    *   **Backend Store** : Base de données managée (AWS RDS PostgreSQL ou Azure SQL) pour la persistance des métadonnées.
    *   **Artifact Store** : Object Storage (S3 ou Azure Blob Storage) pour la durabilité des modèles.
    *   **Serveur** : Déployé sur Kubernetes (EKS/AKS) ou ECS pour la haute disponibilité (plusieurs réplicas).

### Sécurité du Model Registry
*   **Accès** :
    *   Actuellement : Pas d'authentification (réseau privé Docker).
    *   Cible : Mise en place d'un proxy d'authentification (Nginx avec Basic Auth ou OAuth2) devant MLflow, ou utilisation de MLflow Authentication (disponible depuis MLflow 2.0).
*   **Promotion** : Seuls les Lead MLOps ou les pipelines CI/CD (via Service Account) ont les droits d'écriture pour transitionner un modèle vers `Production`. Les Data Scientists ont des droits de lecture/écriture en `Staging` uniquement.

---

## 2. API REST & Performance

### Validation Pydantic et Latence
*   **Impact** : Pydantic V2 (utilisé ici) est écrit en Rust et extrêmement performant. L'overhead de validation est négligeable (< 1ms) par rapport au temps d'inférence du modèle (souvent 10-100ms).
*   **Bénéfice** : La validation stricte protège le modèle contre des entrées mal formées (types incorrects, valeurs manquantes) qui pourraient causer des plantages silencieux ou des prédictions erronées (garbage in, garbage out).

### Chargement du Modèle et Health Check
*   **Démarrage** : Le chargement se fait via le gestionnaire de cycle de vie (`lifespan`) de FastAPI. L'application ne commence à accepter des requêtes qu'une fois le modèle chargé en mémoire.
*   **Health Check** : Endpoint `/health` implémenté.
    *   Vérifie la connexion au serveur MLflow.
    *   Vérifie qu'un modèle est bien chargé en mémoire (`model_cache['model'] is not None`).
    *   Renvoie `503 Service Unavailable` si le modèle est absent, empêchant le routing du trafic par un Load Balancer.

### SLA Cible
*   **Latence** :
    *   p50 : < 50ms (cas nominal).
    *   p95 : < 150ms.
    *   p99 : < 300ms.
    *   Ces valeurs dépendent fortement de la complexité du modèle (Random Forest vs Deep Learning).

---

## 3. CI/CD Pipeline

### Validation de la Qualité des Données
*   **Checks Implémentés** : validation des types et présence des colonnes via Pandas/Pydantic avant l'entraînement.
*   **Checks Avancés (Recommandés)** : Utilisation de librairies comme **Great Expectations** ou **Pandera**.
    *   *Schema Validation* : Vérifier les types de données.
    *   *Value Ranges* : Vérifier que l'âge est > 0, que les probabilités sont entre 0 et 1.
    *   *Null Checks* : Alerter si le taux de valeurs nulles dépasse un seuil.

### Validation des Performances du Modèle
*   **Dataset** : Validation sur un jeu de test maintenu séparément (Golden Dataset) ou un split `test` fixe.
*   **Seuils** :
    *   L'entraînement échoue si `Accuracy < 0.90` (exemple).
    *   Comparaison avec le modèle actuel en production : le nouveau modèle doit être au moins aussi bon (+/- marge).

### Gestion des Secrets (GitHub Actions)
*   Les credentials ne sont **jamais** committés.
*   Injection via **GitHub Action Secrets** (`AWS_ACCESS_KEY_ID`, `DOCKER_REGISTRY_TOKEN`).
*   Utilisation de ces secrets comme variables d'environnement lors de l'exécution des jobs CI/CD.

---

## 4. Monitoring & Observabilité

### Métriques Prometheus
*   **Métriques Métier** :
    *   Nous avons ajouté l'exposition des métriques d'entraînement : `model_training_accuracy`, `f1_score`.
    *   Suivi du volume de prédictions par version de modèle (`predictions_total` label `model_version`).
*   **Drift (Dérive)** :
    *   Pas encore implémenté. Nécessiterait de stocker les distributions des entrées (features) et des sorties (prédictions) pour les comparer à la distribution d'entraînement (Reference vs Current). Outils : Evidently AI ou calcul manuel périodique.

### Qualité en Production
*   Difficile sans "Ground Truth" (la vraie réponse) immédiate.
*   Stratégie : Récupérer le feedback utilisateur ou attendre la réalité terrain (ex: churn réel au bout de 1 mois) pour recalculer les métriques a posteriori.

---

## 5. Grafana Dashboards

### Maintenance et Documentation
*   **Responsable** : L'équipe MLOps/Platform.
*   **Provisioning** : Les dashboards sont **provisionnés via code** (fichiers JSON dans `/monitoring/grafana/dashboards`). Cela permet de les versionner dans Git. Pas de modification manuelle "sauvage" dans l'UI.

### Alertes (Activation)
Les alertes définies (`HighErrorRate`, `HighLatency`, `InstanceDown`) doivent être activées en Production.
*   **Critères** :
    *   Taux d'erreur > 1% pendant 5 min -> PagerDuty/Slack.
    *   Latence p95 > SLA pendant 5 min -> Warning.
    *   Service Down -> Critical immédiat.

---

## 6. Gestion de Projet & Compétences

### Organisation
*   **Data Scientist** : Exploration des données, création et optimisation des modèles (`train.py`), définition des métriques de performance.
*   **ML Engineer** : Packaging du modèle (`api/`), création de l'image Docker, intégration MLflow.
*   **DevOps/Platform** : Infrastructure (Docker Compose/K8s), CI/CD, Monitoring (Prometheus/Grafana).

### Timeline
1.  **Semaine 1** : Setup environnement, pipeline d'entraînement basique, MLflow local.
2.  **Semaine 2** : Développement API, Packaging Docker, Tests unitaires.
3.  **Semaine 3** : Pipeline CI/CD, Monitoring, Déploiement POC.

---

## 7. Tests & Qualité

### Tests Automatisés
*   **Unit Tests** : Tester les fonctions de transformation de données et l'API (codes retour HTTP).
*   **Integration Tests** : Lancer un entraînement complet sur un petit sous-ensemble de données pour vérifier que le pipeline ne plante pas ("Smoke Test").
*   **Performance Tests** : Test de charge avec `Locust` ou `k6` sur l'API pour valider la latence.
*   **Coverage** : Viser > 80% sur le code de l'API et les utilitaires.

### Contrôle Qualité Code
*   **Linters** : `flake8`, `pylint`.
*   **Formatters** : `black`, `isort`.
*   **Sécurité** : `Safety` (check dépendances vulnérables), `Trivy` (scan image Docker), `SonarQube`.

---

## 8. Production Readiness

### Gestion de la Dégradation Modèle
*   Monitoring des métriques techniques (latence, erreurs) et métiers (drift).
*   **Rollback** : Manuel d'abord (via MLflow UI). Automatique si l'API crash au démarrage (CrashLoopBackOff géré par l'orchestrateur).

### Logs
*   Centralisation indispensable. Stack ELK (Elasticsearch, Logstash, Kibana) ou PLG (Promtail, Loki, Grafana).
*   Actuellement : Logs Docker locaux (`json-file`), insuffisant pour la prod.

### Scaling
*   **Horizontal** : L'API est "Stateless" (le modèle est chargé en mémoire). Elle peut scaler horizontalement (ajouter des replicas `mlops-api`) derrière un Load Balancer (Nginx/Traefik/Ingress K8s).

---

## 9. Sécurité & Compliance

### Authentification API
*   Implémenter OAuth2 / OpenID Connect (OIDC) ou simple API Key pour les clients.
*   Utiliser un composant d'API Management (Kong, Apigee) ou l'Ingress Controller pour gérer l'auth.

### Chiffrement
*   **En transit** : TLS (HTTPS) partout (API, MLflow UI).
*   **Au repos** : Chiffrement des volumes de stockage (S3, Disques EBS) et de la base de données MLflow.

### Traçabilité
*   MLflow enregistre automatiquement : *Qui* (User), *Quoi* (Git Commit Hash), *Quand* (Timestamp), *Paramètres*.
*   Git assure la traçabilité du code d'infrastructure et de l'API.
