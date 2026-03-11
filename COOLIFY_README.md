# Déployer Tandemn Tuna sur Coolify

Ce guide explique comment déployer Tandemn Tuna en tant que service Docker Compose sur Coolify.

## Prérequis

1. **Instance Coolify** fonctionnelle
2. **Comptes fournisseurs** : Au moins un fournisseur serverless (Modal, RunPod, etc.)
3. **Credentials AWS/GCP** : Pour les instances spot (optionnel si mode serverless-only)

## Configuration

### 1. Préparation des variables d'environnement

Copiez `.env.example` vers `.env` et configurez vos variables :

```bash
cp .env.example .env
```

Variables essentielles :
- `TUNA_MODEL` : Modèle à déployer (ex: `Qwen/Qwen3-0.6B`)
- `TUNA_GPU` : Type de GPU (ex: `L4`, `A100`)
- `TUNA_PROVIDER` : Fournisseur serverless (`modal`, `runpod`, etc.)
- API keys des fournisseurs choisis

### 2. Configuration Coolify

Dans Coolify, créez une nouvelle application :

1. **Type** : Docker Compose
2. **Source** : Git repository ou upload des fichiers
3. **Variables d'environnement** :
   - Importez votre fichier `.env` ou configurez manuellement
   - Assurez-vous que les credentials sont bien définis

### 3. Modes de déploiement

#### Mode Serverless-only (recommandé pour débuter)
```env
TUNA_SERVERLESS_ONLY=true
```
- Plus simple à configurer
- Pas besoin de credentials AWS/GCP
- Coût plus élevé mais démarrage rapide

#### Mode Hybrid (Serverless + Spot)

**AWS Spot :**
```env
TUNA_SERVERLESS_ONLY=false
TUNA_SPOTS_CLOUD=aws
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

**GCP Spot :**
```env
TUNA_SERVERLESS_ONLY=false
TUNA_SPOTS_CLOUD=gcp
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
```

Avantages :
- Plus économique pour les charges importantes (3-5x moins cher)
- Failover automatique vers serverless
- Support AWS et GCP spot instances

## Fournisseurs supportés

### Modal (recommandé)
```env
TUNA_PROVIDER=modal
```
- Configuration via `modal token new`
- Démarrage rapide, cache automatique

### RunPod
```env
TUNA_PROVIDER=runpod
RUNPOD_API_KEY=your_api_key
```

### Google Cloud Run
```env
TUNA_PROVIDER=cloudrun
GCP_PROJECT_ID=your-project
```

### Autres
- Baseten, Azure Container Apps, Cerebrium également supportés

## Ports exposés

- **8080** : Endpoint principal (OpenAI-compatible)
- **8080/router/health** : Health check

## Utilisation

Une fois déployé, l'API est compatible OpenAI :

```bash
curl http://your-coolify-domain:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Monitoring

- **Status** : `GET /router/health`
- **Logs** : Via interface Coolify
- **Coûts** : Visible dans les dashboards des fournisseurs

## Dépannage

### Problèmes courants

1. **Service ne démarre pas** :
   - Vérifiez les variables d'environnement
   - Validez les API keys avec `tuna check --provider <provider>`

2. **Timeout de déploiement** :
   - Les modèles lourds prennent du temps à télécharger
   - Surveillez les logs pour le progrès

3. **Erreur de credentials** :
   - Mode serverless-only : seules les API keys des fournisseurs sont requises
   - Mode hybrid : ajoutez AWS/GCP credentials

### Commandes de debug

```bash
# Depuis le container
docker-compose exec tuna tuna status --service-name tuna-service
docker-compose exec tuna tuna list
docker-compose logs -f tuna

# Vérifier la configuration GCP
docker-compose exec tuna gcloud auth list
docker-compose exec tuna gcloud config list
docker-compose exec tuna sky check gcp
```

### Configuration GCP spécifique

1. **Quota GPU** : Vérifiez le quota pour VMs préemptibles :
   - Console GCP → IAM & Admin → Quotas
   - Recherchez "Preemptible" + type de GPU + région

2. **APIs requises** :
   - Compute Engine API
   - Cloud Resource Manager API
   - Service Usage API

3. **Permissions** : Le service account doit avoir :
   - Compute Instance Admin
   - Service Account User

## Sécurité

- Les credentials sont montés en lecture seule
- Service accessible uniquement via Coolify proxy
- Utilisateur non-root dans le container

## Scaling

Le scaling est géré automatiquement par :
- **Serverless** : Scaling automatique par le fournisseur
- **Spot instances** : AutoScaling SkyPilot basé sur la charge
- **Router** : Load balancing intelligent entre backends