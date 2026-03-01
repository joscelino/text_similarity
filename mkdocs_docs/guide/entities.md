# Entidades Suportadas

Na etapa de NLP, normalizamos entidades cruciais para que palavras soltas não estraguem o real significado de uma "Medida" ou "Gasto".

## Extractors Nativos

1. **Moeda (`money`)**
   - Transforma `R$ 50,00`, `cinquenta reais`, `50 BRL` todos para a tag neural `<money:50.0>`
2. **Datas (`date`)**
   - Captura conversas de chat (`hoje`, `amanhã`) e transforma em data de calendário sólida ISO `YYYY-MM-DD`.
3. **Dimensões e Unidades (`dimension`)**
   - Lida puramente com valores fracionados que acompanham unidades oficiais (ex: `25.5 kg`, `15cm`).
4. **Números Ordinais/Cardinais (`number`)**
   - Mapeamento estático PT-BR para resolver `"mil", "duas"` em algarismos limpos puramente matemáticos.
5. **Modelos de Tecnologias (`product_model`)**
   - Conserva modelos puros de perda gramatical (`S22`, `XJ-900`) para a IA notar quando há comparação de hardware idêntica.
