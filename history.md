# Historique du projet Tandemn Tuna - Docker Compose & Coolify

## Date: 2026-03-10

### Contexte
L'utilisateur souhaitait déployer Tandemn Tuna en tant que service Docker Compose sur une instance Coolify, avec support complet pour les instances spot GCP.

### Problématique initiale
- Comment containériser Tandemn Tuna pour Coolify
- Configuration des dépendances cloud (AWS CLI, gcloud)
- Support des instances spot GCP en plus d'AWS
- Variables d'environnement et configuration flexible

### Solutions implémentées

#### 1. Dockerfile optimisé
- **Base**: Python 3.11-slim
- **Cloud CLIs**: AWS CLI v2 + Google Cloud SDK complet
- **Dépendances Python**: Installation complète avec `.[all]` + bibliothèques GCP
- **Sécurité**: Utilisateur non-root (`tuna`)
- **Optimisations**: Nettoyage des caches apt, installation via méthodes officielles

#### 2. Docker Compose configuré pour Coolify
- **Port**: 8080 (router Tuna)
- **Variables d'environnement**: Support AWS, GCP, tous les providers serverless
- **Volumes**: Montage des credentials clouds + données persistantes
- **Healthcheck**: Endpoint `/router/health`
- **Command**: Déploiement automatique avec configuration flexible
- **Network**: Réseau dédié `tuna-network`

#### 3. Configuration environnement flexible
- **Modes**: Serverless-only OU Hybrid (serverless + spot)
- **Clouds spot**: AWS ou GCP
- **Providers serverless**: Modal, RunPod, Cloud Run, Baseten, Azure, Cerebrium
- **Variables configurables**: Model, GPU, région, scaling, etc.

#### 4. Support GCP complet
- **CLI**: Installation officielle google-cloud-cli
- **Bibliothèques**: google-cloud-storage, compute, auth
- **Configuration**: Variables GCP_PROJECT_ID, GCP_REGION
- **Documentation**: Quotas, APIs, permissions requises

### Fichiers créés
1. **`Dockerfile`** - Image Docker complète avec toutes dépendances
2. **`docker-compose.yml`** - Service Coolify-ready avec variables d'env
3. **`.env.example`** - Template de configuration avec tous les providers
4. **`COOLIFY_README.md`** - Guide de déploiement détaillé
5. **`history.md`** - Ce fichier d'historique
6. **`CLAUDE.md`** - Documentation technique standard

### Architecture finale
```
Coolify Instance
├── Docker Compose
│   └── Tandemn Tuna Container
│       ├── AWS CLI (spot instances)
│       ├── gcloud CLI (spot GCP)
│       ├── Python + Tuna[all]
│       └── Router (port 8080)
├── Variables d'environnement
│   ├── Cloud credentials
│   ├── Provider API keys
│   └── Configuration modèle
└── Volumes persistants
    ├── Credentials clouds
    └── Données Tuna
```

### Résultats obtenus
- ✅ Container Docker fonctionnel avec toutes les dépendances
- ✅ Support complet AWS spot et GCP spot instances  
- ✅ Configuration flexible via variables d'environnement
- ✅ Guide de déploiement Coolify détaillé
- ✅ Healthchecks et monitoring intégrés
- ✅ Sécurité (utilisateur non-root, volumes read-only)

### Configuration recommandée pour production
```env
# Mode Hybrid GCP pour coûts optimisés
TUNA_SERVERLESS_ONLY=false
TUNA_SPOTS_CLOUD=gcp
TUNA_PROVIDER=modal
TUNA_MODEL=Qwen/Qwen3-0.6B
TUNA_GPU=L4
GCP_PROJECT_ID=votre-projet
GCP_REGION=us-central1
```

### Prochaines étapes possibles
- Tests de charge et monitoring
- Intégration CI/CD pour déploiements automatiques
- Scaling horizontal avec load balancer
- Métriques et alerting avancés

### Lessons learned
- L'installation officielle de gcloud CLI est plus stable que via curl
- Les variables d'environnement doivent être flexibles pour différents use cases
- La documentation est cruciale pour l'adoption en production
- Le support multi-cloud nécessite une attention particulière aux credentials