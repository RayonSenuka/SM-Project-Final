## ============================================================
## Task 3 — Dataset Understanding and Descriptive Analysis
## IT3081 Statistical Modelling — Student Attendance and
## Performance Analysis
## ============================================================

library(dplyr)
library(ggplot2)
library(corrplot)
library(psych)
library(naniar)

## ------------------------------------------------------------
## 0. Load data and rename columns to short, working names
## ------------------------------------------------------------

df_raw <- read.csv("data/Students_Performance_dataset.csv",
                    check.names = FALSE, stringsAsFactors = FALSE)

col_map <- c(
  "University Admission year"                                                = "admission_year",
  "Gender"                                                                   = "gender",
  "Age"                                                                      = "age",
  "H.S.C passing year"                                                       = "hsc_year",
  "Program"                                                                  = "program",
  "Current Semester"                                                        = "semester",
  "Do you have meritorious scholarship ?"                                    = "scholarship",
  "Do you use University transportation?"                                   = "uses_transport",
  "How many hour do you study daily?"                                       = "study_hours",
  "How many times do you seat for study in a day?"                         = "study_sessions",
  "What is your preferable learning mode?"                                  = "learning_mode",
  "Do you use smart phone?"                                                 = "has_smartphone",
  "Do you have personal Computer?"                                          = "has_pc",
  "How many hour do you spent daily in social media?"                       = "social_media_hours",
  "Status of your English language proficiency"                             = "english_proficiency",
  "Average attendance on class"                                             = "attendance",
  "Did you ever fall in probation?"                                         = "probation",
  "Did you ever got suspension?"                                            = "suspension",
  "Do you attend in teacher consultancy for any kind of academical problems?" = "teacher_consultancy",
  "What are the skills do you have ?"                                       = "skills",
  "How many hour do you spent daily on your skill development?"             = "skill_dev_hours",
  "What is you interested area?"                                            = "interested_area",
  "What is your relationship status?"                                       = "relationship_status",
  "Are you engaged with any co-curriculum activities?"                      = "co_curricular",
  "With whom you are living with?"                                          = "living_with",
  "Do you have any health issues?"                                          = "health_issues",
  "What was your previous SGPA?"                                            = "previous_sgpa",
  "Do you have any physical disabilities?"                                  = "physical_disability",
  "What is your current CGPA?"                                              = "current_cgpa",
  "How many Credit did you have completed?"                                 = "credits_completed",
  "What is your monthly family income?"                                     = "family_income"
)

df <- df_raw
colnames(df) <- col_map[colnames(df_raw)]

## ============================================================
## 3.1 Dataset description
## ============================================================

cat("Rows:", nrow(df), " Columns:", ncol(df), "\n")
str(df)
head(df)

## ============================================================
## 3.2 Variable descriptions
## ============================================================
## Print each column's type and number of unique values — a quick
## way to eyeball which variables are numeric vs categorical before
## you write the variable-description table for your report.

var_summary <- data.frame(
  variable = names(df),
  type     = sapply(df, class),
  n_unique = sapply(df, function(x) length(unique(x)))
)
print(var_summary, row.names = FALSE)

## ============================================================
## 3.3 Data quality assessment
## ============================================================

## Issue 1: 'attendance' is text, because one row has a range like "94-98"
## instead of a single number.
class(df$attendance)                              # will print "character"
df$attendance[!grepl("^[0-9]+$", df$attendance)]  # shows the offending value(s)

## Fix: replace a "low-high" range with its midpoint, then convert to numeric
parse_attendance <- function(x) {
  vapply(x, function(v) {
    if (grepl("^[0-9]+\\s*-\\s*[0-9]+$", v)) {
      parts <- as.numeric(strsplit(v, "-")[[1]])
      mean(parts)
    } else {
      as.numeric(v)
    }
  }, numeric(1), USE.NAMES = FALSE)
}
df$attendance <- parse_attendance(df$attendance)
summary(df$attendance)

## Issue 2: current_cgpa and previous_sgpa contain 0 values that are
## implausible for an enrolled student — almost certainly unreported
## data coded as zero, not genuine 0.00 CGPAs.
sum(df$current_cgpa == 0, na.rm = TRUE)
sum(df$previous_sgpa == 0, na.rm = TRUE)

## Fix: treat these as missing rather than real zeros
df$current_cgpa[df$current_cgpa == 0]   <- NA
df$previous_sgpa[df$previous_sgpa == 0] <- NA

## Issue 3: 'program' is constant across every row — confirm, then
## exclude it from any grouping/ANOVA analysis later.
table(df$program)

## Duplicate rows check
sum(duplicated(df))

## ============================================================
## 3.4 Missing value analysis
## ============================================================

colSums(is.na(df))

## Visual missingness map (naniar package)
vis_miss(df)

## Percentage missing per column, sorted
miss_pct <- sort(colMeans(is.na(df)) * 100, decreasing = TRUE)
round(miss_pct[miss_pct > 0], 2)

## ============================================================
## 3.5 Outlier detection
## ============================================================

numeric_cols <- c("age", "study_hours", "study_sessions", "social_media_hours",
                   "skill_dev_hours", "attendance", "previous_sgpa",
                   "current_cgpa", "credits_completed", "family_income")

## Boxplots for all numeric variables in one panel
par(mfrow = c(2, 5))
for (col in numeric_cols) {
  boxplot(df[[col]], main = col, col = "lightblue")
}
par(mfrow = c(1, 1))

## IQR-based outlier flags (Tukey's rule) — prints how many outliers
## per numeric variable
for (col in numeric_cols) {
  q <- quantile(df[[col]], probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  lower <- q[1] - 1.5 * iqr
  upper <- q[2] + 1.5 * iqr
  n_out <- sum(df[[col]] < lower | df[[col]] > upper, na.rm = TRUE)
  cat(sprintf("%-20s outliers: %d\n", col, n_out))
}

## family_income is heavily right-skewed (4,000 to 2,000,000) — a log
## transform is worth considering before using it in regression
ggplot(df, aes(x = family_income)) +
  geom_histogram(bins = 40, fill = "steelblue") +
  labs(title = "Monthly Family Income (raw)", x = "Income", y = "Count")

ggplot(df, aes(x = log1p(family_income))) +
  geom_histogram(bins = 40, fill = "darkgreen") +
  labs(title = "Monthly Family Income (log-transformed)", x = "log(Income + 1)", y = "Count")

## ============================================================
## 3.6 Descriptive statistics
## ============================================================

## Full numeric summary table (mean, sd, median, min, max, skew, kurtosis)
describe(df[, numeric_cols])

## Simpler base-R summary if you prefer
summary(df[, numeric_cols])

## Categorical frequency tables for key grouping variables
for (col in c("gender", "scholarship", "probation", "suspension",
              "learning_mode", "english_proficiency")) {
  cat("\n--", col, "--\n")
  print(table(df[[col]]))
}

## ============================================================
## 3.7 Visualizations
## ============================================================

## Histogram of CGPA
ggplot(df, aes(x = current_cgpa)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Current CGPA", x = "CGPA", y = "Count") +
  theme_minimal()

## Scatter: attendance vs CGPA with trend line — headline chart
ggplot(df, aes(x = attendance, y = current_cgpa)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "darkred") +
  labs(title = "Attendance vs. Current CGPA",
       x = "Average attendance (%)", y = "CGPA") +
  theme_minimal()

## Boxplot: CGPA by probation history
ggplot(df, aes(x = probation, y = current_cgpa, fill = probation)) +
  geom_boxplot() +
  labs(title = "CGPA by Probation History", x = "Ever on probation?", y = "CGPA") +
  theme_minimal() + theme(legend.position = "none")

## Boxplot: CGPA by scholarship status
ggplot(df, aes(x = scholarship, y = current_cgpa, fill = scholarship)) +
  geom_boxplot() +
  labs(title = "CGPA by Scholarship Status", x = "Has scholarship?", y = "CGPA") +
  theme_minimal() + theme(legend.position = "none")

## Bar chart: probation rate by scholarship status
df %>%
  group_by(scholarship) %>%
  summarise(probation_rate = mean(probation == "Yes", na.rm = TRUE)) %>%
  ggplot(aes(x = scholarship, y = probation_rate, fill = scholarship)) +
  geom_col() +
  labs(title = "Probation Rate by Scholarship Status",
       x = "Has scholarship?", y = "Proportion ever on probation") +
  theme_minimal() + theme(legend.position = "none")

## Correlation heatmap across numeric variables
corr_matrix <- cor(df[, numeric_cols], use = "pairwise.complete.obs")
corrplot(corr_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7, tl.col = "black",
         title = "Correlation Heatmap (numeric variables)", mar = c(0, 0, 2, 0))

## ============================================================
## 3.8 Initial business insights
## ============================================================
## Fill this section in AFTER inspecting the actual output above —
## don't finalize the wording until you've seen the real numbers.
## Starting hypotheses to confirm or refute:
##   - Attendance and CGPA show a positive association (check the
##     scatter plot's trend line and the correlation coefficient)
##   - Students with a probation history likely show lower current
##     CGPA (check the boxplot)
##   - Scholarship holders may show both higher CGPA and lower
##     probation rates (check both the boxplot and bar chart)

cor.test(df$attendance, df$current_cgpa)   # formal test of the headline relationship

## ============================================================
## Save the cleaned dataset for use in later tasks (Task 4, 5, ...)
## ============================================================

write.csv(df, "data/Students_Performance_dataset_clean_R.csv", row.names = FALSE)
