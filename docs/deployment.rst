Deployment
==========

Kubernetes + Helm artifacts live under `deployment/`.

Kimi-K2.5 on GKE:

- Standard clusters: `docs/KIMI_K25_GKE_STANDARD.md`
- Autopilot clusters: `docs/KIMI_K25_GKE_AUTOPILOT.md`

Helm chart:

.. code-block:: bash

   helm upgrade --install stateset-agents deployment/helm/stateset-agents \
     --namespace stateset-agents

