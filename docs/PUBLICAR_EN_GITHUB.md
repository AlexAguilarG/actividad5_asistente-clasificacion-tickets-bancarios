# Publicar Actividad5 en GitHub

## Opción A. Repositorio nuevo

1. En GitHub, crea un repositorio llamado `Actividad5` (privado mientras se revisan los términos de la variante traducida de los datos).
2. No agregues un README ni un `.gitignore` desde GitHub, porque ambos ya existen localmente.
3. Descomprime el paquete y abre una terminal dentro de la carpeta `Actividad5`.
4. Ejecuta:

```bash
git init
git branch -M main
git add README.md CHANGELOG.md requirements.txt .gitignore fuentes docs results datos evidencias modelos
git status
git commit -m "feat: estructura reproducible de Actividad 5 con MLflow"
git remote add origin https://github.com/USUARIO/Actividad5.git
git push -u origin main
```

Reemplaza `USUARIO` y la URL por la dirección real del repositorio que creaste. Comprueba `git status` antes de confirmar: los CSV descargados, los embeddings, la carpeta `mlruns/`, los modelos serializados y el entorno virtual deben seguir excluidos por `.gitignore`.

## Opción B. Añadir como carpeta al repositorio anterior

Si deseas conservar la continuidad con el repositorio indicado en tu Actividad 4, puedes clonar `https://github.com/AlexAguilarG/asistente-clasificacion-tickets-bancarios.git` (si sigue siendo el repositorio que usarás), copiar la carpeta `Actividad5/` dentro de ese repositorio y luego ejecutar desde la raíz:

```bash
git add Actividad5/
git status
git commit -m "feat: agrega Actividad 5 - comparación de clasificadores y MLflow"
git push
```

## Seguridad, datos y licencia

- No subas tickets reales, información de clientes, credenciales, tokens ni archivos `.env`.
- Los datos descargados se obtienen ejecutando `python fuentes/datos_prep.py`; no se incluyen completos en el repositorio.
- Verifica los permisos de la variante traducida antes de publicar el repositorio como público.
- No agregues capturas ficticias: guarda solo capturas reales del panel MLflow después de ejecutar `python fuentes/train.py`.
