import os
import io
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv

from agente_orquestrador import OrquestradorAgentes

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN não foi encontrado no .env")

CANAL_BOT = "amigo-natureza"

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
orc = OrquestradorAgentes()


def extrair_resposta_discord(saida: str) -> str:
    if not saida:
        return "Não consegui gerar uma resposta no momento."

    linhas = saida.splitlines()
    idx_resp = None

    for i, linha in enumerate(linhas):
        if "resposta" in linha.lower():
            idx_resp = i
            break

    if idx_resp is None:
        limpa = []
        for linha in linhas:
            l = linha.strip()
            if not l:
                limpa.append(linha)
                continue
            if set(l) == {"="}:
                continue
            if "agente de" in l.lower():
                continue
            if "direcionando para" in l.lower():
                continue
            if "mapas encontrados" in l.lower():
                continue
            if "deseja visualizar algum mapa" in l.lower():
                continue
            limpa.append(linha)

        resposta = "\n".join(limpa).strip()
        return resposta or "Não consegui gerar uma resposta no momento."

    resto = linhas[idx_resp + 1:]

    while resto and not resto[0].strip():
        resto = resto[1:]

    corte_idx = None
    for j, linha in enumerate(resto):
        l = linha.strip().lower()

        if l and set(l) == {"="} and len(l) >= 3:
            corte_idx = j
            break

        if "deseja visualizar algum mapa" in l:
            corte_idx = j
            break

        if "mapas encontrados" in l:
            corte_idx = j
            break

    if corte_idx is not None:
        resto = resto[:corte_idx]

    resposta = "\n".join(resto).strip()
    return resposta or "Não consegui gerar uma resposta no momento."


@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user}")
    print(f"Usando o canal #{CANAL_BOT} em todos os servidores disponíveis.")

    for guild in bot.guilds:
        canal = discord.utils.get(guild.text_channels, name=CANAL_BOT)

        if canal is None:
            try:
                canal = await guild.create_text_channel(CANAL_BOT)
                print(f"Canal #{CANAL_BOT} criado em '{guild.name}'")
                await canal.send(
                    "Olá! Sou o assistente virtual do Parque Nacional da Tijuca.\n"
                    "Faça perguntas sobre clima, trilhas ou informações gerais."
                )
            except discord.Forbidden:
                print(
                    f"Sem permissão para criar canais em '{guild.name}'. "
                    f"Crie manualmente um canal com o nome #{CANAL_BOT}."
                )
        else:
            print(f"O canal #{CANAL_BOT} já existe em '{guild.name}'")


@bot.command(name="limpar")
async def limpar(ctx):
    if isinstance(ctx.channel, discord.TextChannel) and ctx.channel.name != CANAL_BOT:
        await ctx.send(f"Use este comando apenas em #{CANAL_BOT}.")
        return

    orc.limpar_historico()
    await ctx.send("Histórico de conversa limpo.")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if not isinstance(message.channel, discord.TextChannel):
        await bot.process_commands(message)
        return

    if message.channel.name != CANAL_BOT:
        return

    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    pergunta = message.content.strip()
    if not pergunta:
        return

    aguardando = await message.channel.send("Processando sua pergunta...")

    buffer = io.StringIO()
    stdout_original = sys.stdout
    sys.stdout = buffer

    try:
        orc.processar_pergunta(pergunta)
    finally:
        sys.stdout = stdout_original

    bruto = buffer.getvalue()
    resposta = extrair_resposta_discord(bruto)

    try:
        await aguardando.delete()
    except Exception:
        pass

    if len(resposta) <= 2000:
        await message.channel.send(resposta)
    else:
        partes = [resposta[i:i + 1900] for i in range(0, len(resposta), 1900)]
        for parte in partes:
            await message.channel.send(parte)


bot.run(DISCORD_TOKEN)
