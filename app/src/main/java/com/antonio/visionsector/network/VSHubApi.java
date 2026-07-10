package com.antonio.visionsector.network;

import com.antonio.visionsector.network.dto.ModuleBundleDTO;
import java.util.List;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;


public interface VSHubApi {

    // Lista moduli
    @GET("api/v1/modules")
    Call<List<ModuleBundleDTO>> getAllModules();

    @GET("api/v1/modules/{id}/bundle")
    Call<ModuleBundleDTO> getModuleBundle(@Path("id") Long moduleId);

}
