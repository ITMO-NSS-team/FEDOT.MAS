#!/bin/zsh
# Поднимает стенд с внешним адресом: сервер в публичном режиме на порту 4174
# плюс SSH-туннель. Провайдер выбирается переменной TUNNEL (по умолчанию pinggy).
#
#   ./gui/serve-public.sh              — поднять и держать
#   TUNNEL=lhr ./gui/serve-public.sh   — через localhost.run
#   Ctrl+C                             — погасить всё
#
# Адрес не зашит: скрипт печатает тот, что выдал туннель, и повторяет печать,
# если туннель переподключился и имя сменилось.
#
# Что важно понимать про публичный режим: серверный ключ из .env не используется
# вовсе, ключ провайдера вводит сам пользователь в интерфейсе. Токен доступа
# хранится в gui/.access_token и не меняется между запусками — поэтому ссылку,
# которую вы раздали коллегам, не придётся рассылать заново.

set -u

ROOT="${0:A:h:h}"                    # корень репозитория (на уровень выше gui/)
GUI="$ROOT/gui"
# Интерпретатор: окружение репозитория, если оно собрано, иначе тот, что в PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/FEDOT.MAS/.venv/bin/python" ]]; then
  PYTHON="$ROOT/FEDOT.MAS/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
PORT="${GUI_PORT:-4174}"             # 4173 остаётся за локальным стендом
# Провайдер туннеля. pinggy по умолчанию: адрес держится всю сессию. У localhost.run адрес меняется
# прямо посреди живого соединения (наблюдали дважды), у serveo — обрыв запросов
# длиннее ~6 секунд. Плата за pinggy: бесплатная сессия живёт 60 минут, поэтому
# поднимать её лучше перед самым показом.
PROVIDER="${TUNNEL:-pinggy}"        # pinggy | lhr | serveo
SUBDOMAIN="${SERVEO_SUBDOMAIN:-fedotmas-demo}"
# Ключ зарегистрирован на localhost.run, поэтому подключаемся им, а не под nokey:
# анонимные туннели меняют имя при каждом обрыве, с ключом адрес закреплён за аккаунтом.
LHR_KEY="${LHR_KEY:-$HOME/.ssh/id_ed25519}"
LHR_USER="${LHR_USER:-}"            # пусто = локальное имя пользователя, ключ опознаёт аккаунт
TOKEN_FILE="$GUI/.access_token"
LOG_DIR="${TMPDIR:-/tmp}/fedotmas-public"
mkdir -p "$LOG_DIR"

# Токен переживает перезапуск: иначе каждая перезагрузка ломала бы разосланную ссылку.
if [[ ! -s "$TOKEN_FILE" ]]; then
  /usr/bin/python3 -c 'import secrets; print(secrets.token_urlsafe(18))' > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

if ! command -v "$PYTHON" > /dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  print -u2 "Не найден интерпретатор $PYTHON — соберите окружение: uv sync"
  exit 1
fi

cleanup() {
  print "\nГасим стенд…"
  [[ -n "${WATCH_PID:-}" ]] && kill "$WATCH_PID" 2>/dev/null
  [[ -n "${SSH_PID:-}" ]] && kill "$SSH_PID" 2>/dev/null
  [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# --- сервер -----------------------------------------------------------------
if lsof -ti ":$PORT" > /dev/null 2>&1; then
  print "Сервер на порту $PORT уже поднят — используем его."
else
  print "Запускаем сервер на порту $PORT…"
  ( cd "$GUI" && GUI_PUBLIC=1 GUI_PORT="$PORT" GUI_ACCESS_TOKEN="$TOKEN" \
      "$PYTHON" run.py > "$LOG_DIR/server.log" 2>&1 ) &
  SRV_PID=$!
  for i in {1..90}; do
    curl -s -m 2 "http://127.0.0.1:$PORT/api/status" > /dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -s -m 3 "http://127.0.0.1:$PORT/api/status" > /dev/null 2>&1; then
    print -u2 "Сервер не поднялся. Журнал: $LOG_DIR/server.log"
    tail -20 "$LOG_DIR/server.log" >&2
    exit 1
  fi
fi

print ""
print "  Локально:      http://localhost:$PORT/?t=$TOKEN"
print "  Ключ провайдера вводит тот, кто открыл ссылку. Сервер своего ключа не имеет."
print "  Журналы: $LOG_DIR"
case "$PROVIDER" in
  serveo) print "\n  ВНИМАНИЕ: serveo режет запросы длиннее ~6 с — разбор постановки, ответ"
          print "  одной модели и судья будут падать с 502. Держите TUNNEL=pinggy." ;;
  lhr)    print "\n  ВНИМАНИЕ: localhost.run меняет адрес прямо посреди сессии." ;;
  *)      print "\n  Бесплатная сессия pinggy живёт 60 минут: поднимайте перед самым показом." ;;
esac
print ""

# --- туннель ----------------------------------------------------------------
# Соединение рвётся само по себе, поэтому поднимаем его в цикле. Адрес НЕ зашит
# в скрипт: даже с зарегистрированным ключом localhost.run со временем выдаёт
# новое имя, и зашитая ссылка молча вела бы в никуда. Печатаем то, что выдал
# туннель на самом деле, и повторяем печать после каждого переподключения.
case "$PROVIDER" in
  serveo) DOMAIN_RE='[a-z0-9-]+\.serveousercontent\.com' ;;
  lhr)    DOMAIN_RE='[a-z0-9]+\.lhr\.life' ;;
  # Хост у pinggy трёхуровневый (xxx-1-2-3-4.free.pinggy.net), и выражение покороче
  # цепляло только хвост «run.pinggy-free.link» — такая ссылка вела в никуда.
  *)      DOMAIN_RE='[a-z0-9.-]*pinggy[a-z0-9.-]*\.(net|link)' ;;
esac

# Следим за адресом непрерывно, а не только при старте: localhost.run умеет
# сменить имя, не разрывая соединения, и без этого ссылка молча протухала бы.
watch_domain() {
  local shown="" domain
  while true; do
    domain=$(sed -e 's/\x1b\[[0-9;]*m//g' "$LOG_DIR/tunnel.log" 2>/dev/null \
             | grep -aoE "$DOMAIN_RE" | grep -v '^dashboard\.' | tail -1)
    if [[ -n "$domain" && "$domain" != "$shown" ]]; then
      [[ -n "$shown" ]] && print "\n  $(date '+%H:%M:%S') адрес сменился!"
      print ""
      print "  ВНЕШНИЙ АДРЕС: https://$domain/?t=$TOKEN"
      print ""
      shown="$domain"
    fi
    sleep 3
  done
}

watch_domain &
WATCH_PID=$!

while true; do
  : > "$LOG_DIR/tunnel.log"          # чтобы не подхватить адрес прошлой сессии
  case "$PROVIDER" in
    serveo)
      ssh -T -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
          -R "$SUBDOMAIN:80:127.0.0.1:$PORT" serveo.net >> "$LOG_DIR/tunnel.log" 2>&1 &
      ;;
    lhr)
      local_target="localhost.run"
      [[ -n "$LHR_USER" ]] && local_target="$LHR_USER@localhost.run"
      ssh -T -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
          -i "$LHR_KEY" -o IdentitiesOnly=yes \
          -R "80:127.0.0.1:$PORT" "$local_target" >> "$LOG_DIR/tunnel.log" 2>&1 &
      ;;
    *)
      ssh -T -p 443 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
          -R0:127.0.0.1:$PORT a.pinggy.io >> "$LOG_DIR/tunnel.log" 2>&1 &
      ;;
  esac
  SSH_PID=$!
  wait $SSH_PID
  print "$(date '+%H:%M:%S') туннель разорван, переподключаемся…"
  sleep 3
done
