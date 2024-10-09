# AI-systems-lab1

## База знаний SWI-Prolog
```pl
% Факты -- создания видеоигр
game(star_wars_battlefront_2).
game(need_for_speed_unbond).
game(subway_surf).
game(grid_2).

% факты -- персонажи в играх
character(jake).
character(tricky).
character(lucy).
character(prince_k).
character(speedy).
character(yaz).
character(stormtrooper).
character(darth_vader).
character(clone).
character(droid).

% факты -- транспорт
transport(train).
transport(porsche).
transport(bmw).
transport(nissan).
transport(speeder).
transport(x_wing).
transport(tie_fighter).

% факты -- персонажи в играх
character_in_game(jake, subway_surf).
character_in_game(tricky, subway_surf).
character_in_game(lucy, subway_surf).
character_in_game(prince_k, subway_surf).
character_in_game(speedy, need_for_speed_unbond).
character_in_game(yaz, need_for_speed_unbond).
character_in_game(stormtrooper, star_wars_battlefront_2).
character_in_game(darth_vader, star_wars_battlefront_2).
character_in_game(clone, star_wars_battlefront_2).
character_in_game(droid, star_wars_battlefront_2).

% факты -- транспорт в играх
transport_in_game(train, subway_surf).
transport_in_game(porsche, need_for_speed_unbond).
transport_in_game(porsche, grid_2).
transport_in_game(bmw, need_for_speed_unbond).
transport_in_game(bmw, grid_2).
transport_in_game(nissan, need_for_speed_unbond).
transport_in_game(nissan, grid_2).
transport_in_game(speeder, star_wars_battlefront_2).
transport_in_game(x_wing, star_wars_battlefront_2).
transport_in_game(tie_fighter, star_wars_battlefront_2).

% Правило -- вывести весь транспорт в игре
transports_in_game(Game, Transports) :- setof(Transport, transport_in_game(Transport, Game), Transports).

% Правило -- вывести всех персонажей в игре
characters_in_game(Game, Characters) :-
    setof(Character, character_in_game(Character, Game), Characters).

% Правило -- вывести игру, в которой есть определённый персонаж
games_with_character(Character, Games) :-
    setof(Game, character_in_game(Character, Game), Games).

% Правило -- вывести игру, в которой есть определённый транспорт
games_with_transport(Transport, Games) :-
    setof(Game, transport_in_game(Transport, Game), Games).

% Правило -- вывести список транспорта, который находится в одной игре с
% персонажем
transports_for_character(Character, Transports) :-
    character_in_game(Character, Game),
    setof(Transport, transport_in_game(Transport, Game), Transports).

% Правило -- вывести список игр, в которых нет транспорта
game_without_transport(Game) :-
    game(Game), \+ transport_in_game(_, Game).
```

## Примеры выполнения запросов SWI-Prolog

```
?- consult('C:/projects/ai-systems/lab1.pl').
true.

?- character_in_game(jake, Game).
Game = subway_surf.

?- character_in_game(Character, Game), transport_in_game(Transport, Game), Transport = speeder.
Character = stormtrooper,
Game = star_wars_battlefront_2,
Transport = speeder .

?- character_in_game(Character, Game), \+ transport_in_game(_, Game).
false.

?- games_with_transport(Transport, Games), member(star_wars_battlefront_2, Games).
Transport = speeder,
Games = [star_wars_battlefront_2] .
```

## Пример запросов Онтологии Protege

```
character

transport

game and inverse(transport_in_game) value porsche

game and inverse(character_in_game) value jake

transport and transport_in_game some (game and inverse(character_in_game) value darth_vader)

game and (inverse(character_in_game) some character)

character and character_in_game value subway_surf
```

## Пример работы программы Java

```
Можно ли мне играть за персонажа jake ?
subway_surf

Помоги выбрать игру, чтобы кататься на porsche
need_for_speed_unbond
grid_2

а скажи, в какой игре я могу кататься на porsche и играть за персонажа yaz ?
need_for_speed_unbond

а если я хочу учавствовать в звёздных сражениях на tie_fighter , во что мне следует играть? 
star_wars_battlefront_2
```
