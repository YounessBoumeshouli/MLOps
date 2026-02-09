Dans le cadre de l’évolution de nos systèmes d’intelligence artificielle, vous êtes chargé de développer un service de prédiction basé sur un modèle de Machine Learning. Ce service sera exposé via une API REST et doit répondre à des critères de fiabilité, rapidité et maintenabilité.

Pour gérer l’ensemble du cycle de vie du modèle — entraînement, déploiement et suivi en production — vous mettrez en place une chaîne MLOps complète comprenant :

GitHub Actions : pour automatiser les tests, la validation du code et le déploiement continu.
MLflow : pour suivre les expérimentations, versionner les modèles et gérer le registre des modèles.
Prometheus & Grafana : pour surveiller en temps réel la santé et les performances de l’API, notamment la latence, le taux d’erreurs et la disponibilité.
Les Tâches

MLFlow
Intégrer MLflow Tracking dans le script d’entraînement
Logger paramètres, métriques et artefact modèle à chaque run
Comparer les runs via l’UI ou l’API MLflow
Enregistrer le meilleur modèle dans le Model Registry (nom + version)
Promouvoir de Staging à Production
Déployer uniquement la version marquée Production
Ajouter l'endpoint de prédiction
Chargez votre modèle depuis MLflow au démarrage de l'API
Créez un modèle Pydantic pour la validation des données d'entrée
Testez avec des exemples via l'interface Swagger
Mise en place d’un pipeline CI/CD avec GitHub Actions
Mettre en place un workflow GitHub Actions déclenché à chaque push.
Ajouter une étape de validation de la qualité des données.
Ajouter une étape de validation des performances du modèle.
Mettre en place une étape de contrôle de la qualité du code.
Créer un workflow de déploiement continu après validation.
Construire une image Docker du service (API + modèle).
Versionner l’image Docker associée au modèle déployé.
Monitoring avec Prometheus & Grafana
Exposer des métriques applicatives via l’API (endpoint /metrics).
Configurer Prometheus pour collecter les métriques exposées par l’API.
Connecter Grafana à Prometheus comme source de données.
Créer un dashboard Grafana pour le suivi du service.
Visualiser en temps réel l’état de l’API.
Afficher les métriques clés : nombre de requêtes, latence, erreurs, temps d’inférence
Surveiller l’état du conteneur Docker (CPU, RAM, réseau)
Configurer des alertes sur les métriques critiques (optionnel).
Modalités pédagogiques
Travail : en groupe

Durée : 5 jours ouvrés

Période : Du 26/01/2026 au 30/01/2026 avant minuit.

Modalités d'évaluation
Mise en situation
Code review
Culture du projet
Livrables
Code source : entraînement ML, API FastAPI, Dockerfile, requirements.txt.
Modèle entraîné : fichier versionné + métriques dans MLflow.
Pipeline CI/CD : workflows GitHub Actions pour tests et déploiement Docker.
Docker : image fonctionnelle + instructions pour exécution locale.
Monitoring : endpoint /metrics, dashboard Grafana, configuration Prometheus.
README.md (description, installation, technologies)
Critères de performance
API REST
- Disponible et réactive (latence p95 < 1 s).
- Capable de gérer plusieurs requêtes simultanées.

Pipeline CI/CD
- Tests et validations automatiques à chaque push.
- Déploiement reproductible via Docker.
- Qualité du code vérifiée automatiquement.

Monitoring
- Collecte des métriques essentielles (CPU, RAM, latence, erreurs).
- Dashboard Grafana clair et à jour.
- Alertes sur indisponibilité ou lenteur (optionnel).

Gestion des modèles
- Versioning des modèles avec MLflow.
- Traçabilité des paramètres et métriques.
- Promotion facile du modèle Staging → Production.