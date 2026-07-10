package com.antonio.visionsector.network;

import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {

    private static final String BASE_URL = "http://192.168.178.42:8080/";
    private static Retrofit retrofit = null;

    public static VSHubApi getApi() {
        if (retrofit == null) {

            // Creiamo uno scudo per la nostra connessione
            OkHttpClient client = new OkHttpClient.Builder()
                    .connectTimeout(10, TimeUnit.SECONDS) // Max 10 sec per trovare il server
                    .readTimeout(15, TimeUnit.SECONDS)    // Max 15 sec per scaricare i dati
                    .retryOnConnectionFailure(true)       // STANDARD ROBUSTO: Riprova in automatico se cade la linea!
                    .build();

            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(client) // Agganciamo il nostro client corazzato
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit.create(VSHubApi.class);
    }
}