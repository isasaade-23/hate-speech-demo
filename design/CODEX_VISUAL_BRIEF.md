# Luciola · Briefing visual (para execução pelo Codex)

Direção de arte para evoluir o demo (`streamlit_app.py`) de "sequência de widgets do
Streamlit" para um produto com identidade própria. Este briefing trata de composição,
profundidade e linguagem visual. A estrutura semântica, os estados e a acessibilidade
já existem e devem ser preservados.

---

## 1. O que a Luciola é

Classificador científico de discurso de ódio, bilíngue (EN/PT). Luciola é vaga-lume:
muitas luzes pequenas. Os três pilares visuais da identidade:

1. **Luz**: brilho, halos, pontos luminosos, pulsação.
2. **Linguagem**: palavras, fragmentos, estruturas linguísticas abstratas.
3. **IA**: redes, nós, conexões, fluxo de informação.

Estética alvo: **minimalismo sofisticado, científico e contemporâneo**. Ferramenta de
IA para pesquisa, não notebook convertido em app, não dashboard SaaS genérico.

---

## 2. Paleta (INTOCÁVEL) e hierarquia de uso

Preservar integralmente os valores. O que muda é a disciplina de uso.

| Papel | Light | Dark | Uso |
|---|---|---|---|
| primary (ação/destaque) | `#EE6C4D` coral | idem | CTAs, resultado-chave, marca |
| secondary | `#3D5A80` slate | `#9DB0D2` | apoio, links, labels de seção |
| deep (superfície de impacto) | `#1F3050` | `#1b2740` | hero, painel de veredito |
| accent | `#F4A261` âmbar | `#F4B266` | avisos, badge de incerteza |
| background | `#F6F2EE` bone | `#121a2b` | página |
| surface | `#ffffff` | `#1b2740` | cards |
| surface-2 | `#EFE9E0` | `#16223a` | áreas de apoio (explicabilidade) |
| texto | `#2B2B2B` / heading `#1F3050` | `#E4ECF6` / `#EAF0F8` | corpo/títulos |
| muted | `#5D6673` | `#9DB0D2` | auxiliares |
| success | `#0F7A56` (texto) `#1D9E75` (gráfico) | `#4FC79A` | não-ódio |
| warning | `#8A5A10` (texto) | `#F4B266` | incerteza |
| danger | `#B93A20` (texto) `#EE6C4D` (gráfico) | `#FF8A6B` | ódio |

**Regra 70/25/5.** Cerca de 70% da tela em neutros (bone, surface, muted), 25% em
slate/deep, 5% em coral+âmbar. Coral nunca aparece em dois elementos vizinhos com o
mesmo peso. Se tudo grita, nada grita: cores fortes são reservadas para ação, veredito
e um único destaque por seção.

---

## 3. Superfícies e profundidade

Cinco níveis, do fundo para a frente:

1. `bg` (bone) com a textura temática (seção 4);
2. `bg-textured`: a mesma cor com as partículas/rede visíveis a baixíssimo contraste;
3. `surface`: cards de conteúdo, borda 1px sutil (`--line`), sombra leve difusa
   (ex.: `0 2px 12px rgba(31,48,80,.06)`);
4. `elevated`: veredito e hero, sombra um pouco maior + halo de luz;
5. `accent surface`: o deep navy com gradiente, usado no máximo 2x por página.

**Aposentar o brutalism atual como regra única**: as sombras duras deslocadas
(`8px 8px 0`) e bordas de 3px podem sobreviver como assinatura em 1-2 elementos
(ex.: o número-herói), mas o sistema padrão passa a ser: cantos arredondados
(radius 12-16px em cards, pill 999px em badges/botões de exemplo), bordas 1px,
sombras muito leves, e profundidade por camadas em vez de por peso de borda.
Transparência/glass só pontual (ex.: header sticky com blur), nunca a página toda.

---

## 4. Textura de fundo e background dinâmico

Uma única textura temática, extremamente sutil, cobrindo os três pilares ao mesmo
tempo: **constelação de vaga-lumes que é também um grafo** (pontos pequenos, alguns
com halo suave, ligados por linhas finas quase invisíveis).

Especificação:
- Apenas cores da paleta (slate e coral sobre bone; no dark, os mesmos pontos claros).
- Opacidade máxima dos elementos: 5-6% para linhas, 10-12% para pontos, halos por
  radial-gradient. O texto nunca disputa com o fundo (contraste do conteúdo intacto).
- Implementação leve: SVG inline como background-image (data URI) ou canvas discreto.
  Se animado (drift lento das partículas, pulsação de 2-3 pontos), respeitar
  `prefers-reduced-motion: reduce` desligando o movimento.
- Densidade baixa: é céu de interior de mata à noite, não papel de parede.

---

## 5. Forma e composição

- **Assimetria editorial**: nem toda seção é coluna central + card. Hero em duas
  colunas (mensagem à esquerda, representação gráfica de luz+linguagem à direita);
  resultados como faixa de destaque + métricas menores; metodologia em cards pequenos;
  footer horizontal.
- **Ritmo**: alternar seções largas e estreitas, cards grandes e pequenos, pesos
  tipográficos contrastados. Alinhamento à esquerda por padrão; centralização só em
  momentos de destaque (CTA, número-herói).
- **Quebras de grade controladas**: um badge que vaza o card, um círculo de luz
  parcialmente sobreposto à borda de uma seção, uma linha que conecta dois blocos.
  Sempre ancorado ao grid principal, nunca aleatório.
- **Divisores não convencionais**: em vez de `<hr>`, usar três pontos luminosos, uma
  linha fina com um nó no meio, ou o próprio gradiente do fundo. Divisores decorativos
  levam `role="presentation"`.
- **Whitespace com intenção**: espaço em volta do que importa; eliminar os vãos
  verticais default do Streamlit (gap dos blocos) via CSS, para que o espaço existente
  seja escolhido, não sobra.

---

## 6. Iconografia

- **Uma única família**, estilo outline, stroke consistente (2px), cantos arredondados,
  16/20/24px. Implementar como SVG inline (sprite ou strings Python), NUNCA emoji,
  NUNCA mistura de estilos. Sugestão de referência: Lucide/Feather (MIT), embutidos.
- Ícones são comunicação: entrada de texto, idioma/globo, modelo/chip, classificação,
  confiança/gauge, explicação/lupa, informação, alerta, documentação, código, alvo
  (limiar), rede (transformer), raio (latência), vaga-lume estilizado (marca).
- Cada label de seção pode carregar seu micro-ícone (12-14px, cor muted) como parte
  do sistema de divisórias.

---

## 7. O veredito como linguagem visual própria

O resultado deixa de ser texto num quadrado e vira uma composição:

- **Classificação principal**: ícone + palavra + cor + forma (ex.: selo/pill grande
  com halo na cor do estado). Ódio = danger, não-ódio = success, incerteza = âmbar.
- **Probabilidade**: manter o número grande, acompanhado de **barra horizontal** com
  marcador de limiar (já existe; refinar como gauge/segmento com gradiente sutil).
- **Distribuição entre classes**: barra dupla ou donut minimalista
  (P(ódio) vs P(não-ódio)); só se ajudar a leitura, sem decorar.
- **Incerteza**: quando |score - limiar| < 8 pontos, o estado inteiro muda para a
  família âmbar (cor, ícone, halo), não apenas um badge.
- **Confiança**: indicador de 3 níveis (alta/média/baixa) como micro-infográfico
  (3 pontos preenchidos progressivamente), não só palavra.
- **Explicabilidade**: manter os chips de termos com sinal (já implementados com
  atribuição linear real tfidf×coef). Evoluir: intensidade da cor proporcional ao
  peso, mini-barra dentro do chip, e um callout curto ligando os termos ao veredito.
  Grid de fatos (modelo, latência, idioma, limiar) com micro-ícones.

---

## 8. Metáfora de chat: onde sim, onde não

- **Sim**: a área de interação pode ler-se como diálogo. O texto do usuário aparece
  como mensagem (balão à direita, surface), a análise como resposta da Luciola
  (à esquerda, com a marca pulsando enquanto "analisa"). Isso humaniza o fluxo
  digitar → analisar → responder.
- **Não**: nada de histórico infinito, avatares fofos ou tom de assistente. É UM
  turno: texto → análise. O resto da página continua editorial/científico. A resposta
  contém o veredito visual da seção 7, não texto corrido.

---

## 9. Infográficos de método

- **Pipeline**: TEXT → LANGUAGE → MODEL → CLASSIFICATION → CONFIDENCE → EXPLANATION
  como fluxo visual horizontal (vertical no mobile): nós conectados por linhas finas,
  cada nó com ícone + rótulo de uma palavra. É a ponte perfeita entre "rede" e "luz"
  (o nó ativo pode acender).
- **How it was built** (5 passos atuais): virar blocos conectados com ícones, menos
  parágrafo, mais diagrama.
- **Tecnologias**: faixa de badges/pills consistentes (NLP, multilingual, TF-IDF,
  transformers, McNemar...), mesma família visual dos chips de termos.
- **Identidade científica separada**: conteúdo metodológico (dataset, métricas,
  limitações, tabela, heatmap) tem tratamento visual próprio, mais sóbrio (surface-2,
  tipografia menor, ícone de "método"), distinto do bloco de resultado vivo.

---

## 10. Microinterações e estados

- Estados obrigatórios por componente: default, hover, focus, active, loading,
  success, warning, error, disabled. Focus SEMPRE visível (outline coral já existe).
- Hover: elevação sutil (+2px de sombra), brilho leve na borda, nunca saltos bruscos.
- **Loading da Luciola**: substituir o spinner genérico por pulsação luminosa
  (o vaga-lume da marca ou 3 pontos que acendem em sequência). Streamlit: possível
  via placeholder HTML/CSS animado no card de resultado enquanto computa, em vez de
  `st.spinner`. Com `prefers-reduced-motion`, vira fade estático.
- Transições: 150-250ms, easing suave; a revelação por scroll já existe (IntersectionObserver).
- Ao concluir a classificação, o veredito entra com um "acender" (fade + leve glow),
  reforçando a metáfora.

---

## 11. Hero e seletor de idioma

- **Hero forte**: nome Luciola + proposta + EN/PT + representação gráfica abstrata de
  luz+linguagem+IA (ex.: palavras fragmentadas virando pontos de luz conectados).
  Duas colunas no desktop; a arte nunca compete com o texto (contraste baixo).
- **Seletor EN/PT como identidade**: não parecer widget. Ideia: pill único deslizante
  com as duas siglas e um ponto luminoso que desliza para o idioma ativo. O toggle de
  tema segue o mesmo DNA (sol/lua da mesma família de ícones).

---

## 12. Responsividade dos elementos gráficos

- Textura: menos densa no mobile (ou estática).
- Pipeline/infográficos: reorganizar (horizontal → vertical), não só encolher.
- Chips, badges, gauges: touch targets ≥ 40px; grids de fatos colapsam para 2 colunas
  e depois 1.
- Ordem mobile preservada: Input → Resultado → Explicação.

---

## 13. Anti-padrões (não fazer)

- Não usar todas as cores com a mesma intensidade; não colorir por decoração.
- Não encher a página de cards iguais com sombras fortes (dashboard genérico).
- Não colocar vaga-lume literal em todo lugar; a metáfora é abstrata e transversal.
- Não adicionar elemento visual sem função (estética, informação ou orientação).
- Não deixar textura/partícula competir com texto (baixo contraste sempre).
- Não transformar tudo em chat.
- Não centralizar tudo.
- Nenhum elemento pode parecer de outro sistema: botões, cards, chips, alertas,
  tooltips, métricas e gráficos compartilham radius, stroke, paleta e tipografia.

---

## 14. Restrições técnicas (aprendidas neste código; violar = quebrar)

1. **Um arquivo**: `streamlit_app.py`. CSS no bloco de tokens existente
   (`:root` light + `DARK_CSS` override). Toda cor nova entra como token nos DOIS temas.
2. **`st.markdown` trata linha HTML indentada com 4+ espaços como code block.**
   Usar o helper existente `html_block()` em todo bloco HTML multilinha.
3. `st.markdown` remove `<script>`. JS só via `render_iframe()` (padrão já usado pelo
   heatmap e pelo scroll-reveal). SVG inline sobrevive como data URI em CSS
   background; SVG direto no markdown pode ser sanitizado (testar; senão, data URI).
4. Widgets não podem ser envolvidos por HTML próprio: estilizar containers com chave
   (`st.container(key=...)` → classe `.st-key-<key>`), como o `input_card` atual.
5. **Não alterar**: números e afirmações científicas (0.784/0.729/0.835, p<0.001,
   limiar lido do bundle), textos do dict `T` (EN e PT sempre em paridade; sem
   travessão em texto visível), os 4 exemplos, a lógica de estados
   (idle/analyzing/result/uncertain/error), a atribuição de termos (`top_terms`),
   o heatmap iframe e seus dados, a acessibilidade existente (skip link, aria-live,
   role=alert, focus-visible, headings h1→h2→h3).
6. CPU only, sem dependência JS externa nova; fontes via Google Fonts já importadas
   (Lato). Se adicionar peso/família, justificar e importar no mesmo `@import`.
7. Dark mode: tudo que for desenhado precisa funcionar nos dois temas.
8. Testar com `python -c "import ast; ast.parse(open('streamlit_app.py').read())"` e
   rodando `streamlit run` local antes de concluir.

---

## 15. Critérios de aceite

- A página não parece Streamlit: header, hero, seções, ferramenta e footer têm o
  mesmo DNA visual Luciola.
- Paleta original preservada; coral e âmbar somam ~5% da área visível.
- Existe profundidade perceptível (5 níveis) sem sombra pesada.
- Textura de fundo presente, notada apenas quando se procura.
- Uma única família de ícones em toda a interface; zero emoji.
- O veredito é uma composição visual (ícone+cor+forma+número+infográfico), com
  estados distintos para ódio, não-ódio, incerteza e erro.
- Pipeline de funcionamento visualizado como fluxo conectado.
- Loading temático (pulsação de luz), não spinner genérico.
- Mobile reorganiza, não encolhe.
- Dark mode íntegro. A11y íntegra. Números e textos científicos intocados.
