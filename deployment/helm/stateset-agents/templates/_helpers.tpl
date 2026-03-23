{{- define "stateset-agents.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "stateset-agents.fullname" -}}
{{- .Release.Name -}}
{{- end -}}

{{- define "stateset-agents.image" -}}
{{- $repo := .repository -}}
{{- $tag := .tag | default "latest" -}}
{{- $digest := .digest | default "" -}}
{{- if $digest -}}
{{ printf "%s@%s" $repo $digest }}
{{- else -}}
{{ printf "%s:%s" $repo $tag }}
{{- end -}}
{{- end -}}

{{- define "stateset-agents.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (printf "%s-sa" (include "stateset-agents.fullname" .)) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
