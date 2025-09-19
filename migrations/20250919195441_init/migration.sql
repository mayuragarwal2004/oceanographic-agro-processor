-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "postgis";

-- CreateTable
CREATE TABLE "platforms" (
    "id" TEXT NOT NULL,
    "platform_number" TEXT NOT NULL,
    "platform_type" TEXT,
    "float_serial" TEXT,
    "project_name" TEXT,
    "pi_name" TEXT,
    "institution" TEXT,
    "data_center" TEXT,
    "wmo_inst_type" TEXT,
    "firmware_version" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "platforms_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "profiles" (
    "id" TEXT NOT NULL,
    "platform_id" TEXT NOT NULL,
    "cycle_number" INTEGER NOT NULL,
    "profile_date" TIMESTAMP(3) NOT NULL,
    "direction" TEXT,
    "data_mode" TEXT,
    "latitude" DOUBLE PRECISION NOT NULL,
    "longitude" DOUBLE PRECISION NOT NULL,
    "location" geometry(Point, 4326),
    "geohash" TEXT,
    "position_qc" INTEGER,
    "profile_pres_qc" TEXT,
    "profile_temp_qc" TEXT,
    "profile_psal_qc" TEXT,
    "juld" DOUBLE PRECISION,
    "juld_qc" INTEGER,
    "juld_location" DOUBLE PRECISION,
    "data_state_indicator" TEXT,
    "config_mission_number" INTEGER,
    "positioning_system" TEXT,
    "vertical_sampling_scheme" TEXT,
    "dc_reference" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "profiles_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "measurements" (
    "id" TEXT NOT NULL,
    "profile_id" TEXT NOT NULL,
    "pressure" DOUBLE PRECISION,
    "pressure_qc" INTEGER,
    "pressure_adjusted" DOUBLE PRECISION,
    "pressure_adjusted_qc" INTEGER,
    "pressure_adjusted_error" DOUBLE PRECISION,
    "temp" DOUBLE PRECISION,
    "temp_qc" INTEGER,
    "temp_adjusted" DOUBLE PRECISION,
    "temp_adjusted_qc" INTEGER,
    "temp_adjusted_error" DOUBLE PRECISION,
    "psal" DOUBLE PRECISION,
    "psal_qc" INTEGER,
    "psal_adjusted" DOUBLE PRECISION,
    "psal_adjusted_qc" INTEGER,
    "psal_adjusted_error" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "measurements_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "calibrations" (
    "id" TEXT NOT NULL,
    "platform_number" TEXT NOT NULL,
    "parameter" TEXT NOT NULL,
    "equation" TEXT NOT NULL,
    "coefficient" TEXT NOT NULL,
    "calibration_date" TIMESTAMP(3),
    "comment" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "calibrations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "processing_history" (
    "id" TEXT NOT NULL,
    "profile_id" TEXT NOT NULL,
    "institution" TEXT,
    "step" TEXT,
    "software" TEXT,
    "software_release" TEXT,
    "reference" TEXT,
    "date_update" TIMESTAMP(3) NOT NULL,
    "action" TEXT NOT NULL,
    "parameter" TEXT,
    "start_pressure" DOUBLE PRECISION,
    "stop_pressure" DOUBLE PRECISION,
    "previous_value" DOUBLE PRECISION,
    "qc_test" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "processing_history_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "qc_flags" (
    "id" INTEGER NOT NULL,
    "description" TEXT NOT NULL,
    "meaning" TEXT NOT NULL,

    CONSTRAINT "qc_flags_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ocean_regions" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "boundary" geometry(Polygon, 4326) NOT NULL,
    "description" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ocean_regions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "station_parameters" (
    "id" TEXT NOT NULL,
    "profile_id" TEXT NOT NULL,
    "parameter" TEXT NOT NULL,

    CONSTRAINT "station_parameters_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "platforms_platform_number_key" ON "platforms"("platform_number");

-- CreateIndex
CREATE INDEX "profiles_latitude_longitude_idx" ON "profiles"("latitude", "longitude");

-- CreateIndex
CREATE INDEX "profiles_location_idx" ON "profiles" USING GIST ("location");

-- CreateIndex
CREATE INDEX "profiles_profile_date_idx" ON "profiles"("profile_date");

-- CreateIndex
CREATE INDEX "profiles_geohash_idx" ON "profiles"("geohash");

-- CreateIndex
CREATE UNIQUE INDEX "profiles_platform_id_cycle_number_profile_date_key" ON "profiles"("platform_id", "cycle_number", "profile_date");

-- CreateIndex
CREATE INDEX "measurements_profile_id_idx" ON "measurements"("profile_id");

-- CreateIndex
CREATE INDEX "measurements_pressure_idx" ON "measurements"("pressure");

-- CreateIndex
CREATE INDEX "measurements_temp_idx" ON "measurements"("temp");

-- CreateIndex
CREATE INDEX "measurements_psal_idx" ON "measurements"("psal");

-- CreateIndex
CREATE INDEX "calibrations_platform_number_parameter_idx" ON "calibrations"("platform_number", "parameter");

-- CreateIndex
CREATE INDEX "processing_history_profile_id_idx" ON "processing_history"("profile_id");

-- CreateIndex
CREATE UNIQUE INDEX "ocean_regions_name_key" ON "ocean_regions"("name");

-- CreateIndex
CREATE UNIQUE INDEX "ocean_regions_code_key" ON "ocean_regions"("code");

-- CreateIndex
CREATE INDEX "ocean_regions_boundary_idx" ON "ocean_regions" USING GIST ("boundary");

-- CreateIndex
CREATE UNIQUE INDEX "station_parameters_profile_id_parameter_key" ON "station_parameters"("profile_id", "parameter");

-- AddForeignKey
ALTER TABLE "profiles" ADD CONSTRAINT "profiles_platform_id_fkey" FOREIGN KEY ("platform_id") REFERENCES "platforms"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "measurements" ADD CONSTRAINT "measurements_profile_id_fkey" FOREIGN KEY ("profile_id") REFERENCES "profiles"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "processing_history" ADD CONSTRAINT "processing_history_profile_id_fkey" FOREIGN KEY ("profile_id") REFERENCES "profiles"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "station_parameters" ADD CONSTRAINT "station_parameters_profile_id_fkey" FOREIGN KEY ("profile_id") REFERENCES "profiles"("id") ON DELETE CASCADE ON UPDATE CASCADE;
