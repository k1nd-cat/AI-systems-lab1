package org.lab2;

import org.projog.api.Projog;
import org.projog.api.QueryResult;

import java.io.*;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        while (true) {
            var request = getRequest();
            if (request == null) {
                System.out.println("Ошибка запроса");
                continue;
            }
            var model = strParser(request);
            if (model == null) {
                System.out.println("Ошибка запроса");
                continue;
            }

            var response = getResponse(model);
            System.out.println(response);
        }
    }

    private static String getRequest() {
        var in = new BufferedReader(new InputStreamReader(System.in));
        try {
            return in.readLine();
        } catch (IOException error) {
            return null;
        }
    }

    private static Model strParser(String str) {
        String[] characters = {"jake", "tricky", "lucy", "prince_k", "speedy", "yaz", "stormtrooper", "darth_vader", "clone", "droid"};
        var characterList = Arrays.asList(characters);
        String[] transports = {"train", "porsche", "bmw", "nissan", "speeder", "x_wing", "tie_fighter"};
        var transportList = Arrays.asList(transports);

        var model = new Model();
        var isCharacter = false;
        var isTransport = false;
        for (String word : str.split(" ")) {
            if (characterList.contains(word)) {
                if (isCharacter) return null;
                isCharacter = true;
                model.character = word;
            } else if (transportList.contains(word)) {
                if (isTransport) return null;
                isTransport = true;
                model.transport = word;
            }
        }

        if (!isCharacter && !isTransport) return null;

        return model;
    }

    private static String getResponse(Model model) {
        var projog = new Projog();
        var fileName = "lab1.pl";
        projog.consultFile(new File(fileName));

        QueryResult result;
        if (model.character != null && model.transport == null) {
            result = projog.executeQuery("character_in_game(" + model.character + ", X).");
        } else if (model.character == null && model.transport != null) {
            result = projog.executeQuery("transport_in_game(" + model.transport + ", X).");
        } else {
            result = projog.executeQuery("transport_in_game(" + model.transport + ", X), character_in_game(" + model.character + ", X).");
        }

        var resultStr = new StringBuilder();
        var counter = 0;
        while (result.next()) {
            resultStr.append(result.getAtomName("X")).append("\n");
            counter++;
        }

        if (counter == 0) return "По вашему запросу ничего не найдено";
        else return resultStr.toString();
    }
}
