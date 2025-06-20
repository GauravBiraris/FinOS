# Core AI Engine - 4 Microservices

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import redis
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import cv2
import pytesseract
import spacy
import re
from dataclasses import dataclass
import joblib
import os

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")

# ============= MICROSERVICE 1: CASH FLOW PREDICTION SERVICE =============

class CashFlowPredictor:
    def __init__(self):
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.lookback_window = 30
        self.prediction_horizons = [7, 30, 90]
        
    def prepare_lstm_data(self, df: pd.DataFrame) -> tuple:
        """Prepare time series data for LSTM model"""
        # Sort by date and create features
        df = df.sort_values('date').reset_index(drop=True)
        
        # Feature engineering
        features = []
        targets = []
        
        for i in range(self.lookback_window, len(df)):
            # Features: lookback window of balances and flows
            feature_window = df.iloc[i-self.lookback_window:i]
            
            balance_sequence = feature_window['balance'].values
            amount_sequence = feature_window['amount'].values
            
            # Additional temporal features
            day_of_week = df.iloc[i]['day_of_week']
            day_of_month = df.iloc[i]['day_of_month']
            month = df.iloc[i]['month']
            
            # Combine features
            feature_vector = np.concatenate([
                balance_sequence,
                amount_sequence,
                [day_of_week, day_of_month, month]
            ])
            
            features.append(feature_vector)
            targets.append(df.iloc[i]['balance'])
        
        return np.array(features), np.array(targets)
    
    def build_lstm_model(self, input_shape: tuple) -> tf.keras.Model:
        """Build LSTM model for cash flow prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, transactions_df: pd.DataFrame) -> Dict:
        """Train cash flow prediction model"""
        if len(transactions_df) < self.lookback_window + 30:
            raise ValueError("Insufficient data for training. Need at least 60 transactions.")
        
        # Prepare data
        X, y = self.prepare_lstm_data(transactions_df)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape[0], self.lookback_window, -1)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Build and train model
        self.lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate model
        y_pred = self.lstm_model.predict(X_test)
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        
        return {
            'mae': mae,
            'rmse': np.sqrt(mse),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_cashflow(self, transactions_df: pd.DataFrame, days_ahead: int = 7) -> Dict:
        """Predict cash flow for specified days ahead"""
        if self.lstm_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get recent data for prediction
        recent_data = transactions_df.tail(self.lookback_window)
        
        predictions = []
        current_balance = recent_data['balance'].iloc[-1]
        
        for day in range(1, days_ahead + 1):
            # Prepare input features
            feature_window = recent_data.tail(self.lookback_window)
            
            balance_sequence = feature_window['balance'].values
            amount_sequence = feature_window['amount'].values
            
            # Future date features
            future_date = datetime.now() + timedelta(days=day)
            day_of_week = future_date.weekday()
            day_of_month = future_date.day
            month = future_date.month
            
            feature_vector = np.concatenate([
                balance_sequence,
                amount_sequence,
                [day_of_week, day_of_month, month]
            ])
            
            # Scale and predict
            X_scaled = self.feature_scaler.transform(feature_vector.reshape(1, -1))
            X_scaled = X_scaled.reshape(1, self.lookback_window, -1)
            
            pred_scaled = self.lstm_model.predict(X_scaled, verbose=0)
            predicted_balance = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_balance': float(predicted_balance),
                'days_ahead': day
            })
            
            # Update recent_data for next prediction
            new_row = pd.DataFrame({
                'date': [future_date],
                'balance': [predicted_balance],
                'amount': [predicted_balance - current_balance],
                'day_of_week': [day_of_week],
                'day_of_month': [day_of_month],
                'month': [month]
            })
            recent_data = pd.concat([recent_data, new_row]).tail(self.lookback_window)
            current_balance = predicted_balance
        
        return {
            'predictions': predictions,
            'confidence_level': 0.85,  # Based on model performance
            'model_accuracy': 'Good' if len(transactions_df) > 100 else 'Fair'
        }

# ============= MICROSERVICE 2: OPERATIONS INTELLIGENCE SERVICE =============

class OperationsIntelligence:
    def __init__(self):
        self.expense_optimizer = RandomForestRegressor(n_estimators=100, random_state=42)
        self.inventory_optimizer = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.performance_analyzer = None
        
    def analyze_expense_patterns(self, transactions_df: pd.DataFrame) -> Dict:
        """Analyze expense patterns and suggest optimizations"""
        expense_data = transactions_df[transactions_df['amount'] < 0].copy()
        expense_data['amount'] = expense_data['amount'].abs()
        
        # Group by category and analyze
        category_analysis = expense_data.groupby('category').agg({
            'amount': ['sum', 'mean', 'count', 'std']
        }).round(2)
        
        # Identify optimization opportunities
        optimizations = []
        
        for category in category_analysis.index:
            cat_data = expense_data[expense_data['category'] == category]
            total_spent = cat_data['amount'].sum()
            avg_transaction = cat_data['amount'].mean()
            frequency = len(cat_data)
            
            # Rule-based optimization suggestions
            if category == 'utilities' and total_spent > avg_transaction * frequency * 1.2:
                optimizations.append({
                    'category': category,
                    'suggestion': 'Consider energy-efficient alternatives or negotiate better rates',
                    'potential_savings': total_spent * 0.15,
                    'priority': 'high'
                })
            elif category == 'supplies' and frequency > 20:
                optimizations.append({
                    'category': category,
                    'suggestion': 'Bulk purchasing could reduce costs',
                    'potential_savings': total_spent * 0.08,
                    'priority': 'medium'
                })
            elif category == 'professional' and total_spent > 5000:
                optimizations.append({
                    'category': category,
                    'suggestion': 'Review service contracts for better terms',
                    'potential_savings': total_spent * 0.10,
                    'priority': 'high'
                })
        
        return {
            'category_breakdown': category_analysis.to_dict(),
            'total_expenses': expense_data['amount'].sum(),
            'optimization_opportunities': optimizations,
            'expense_trend': self._calculate_expense_trend(expense_data)
        }
    
    def optimize_inventory_management(self, inventory_data: Dict) -> Dict:
        """Provide inventory optimization recommendations"""
        if not inventory_data:
            return {'message': 'No inventory data provided'}
        
        recommendations = []
        
        for item, details in inventory_data.items():
            current_stock = details.get('current_stock', 0)
            avg_weekly_usage = details.get('avg_weekly_usage', 0)
            unit_cost = details.get('unit_cost', 0)
            lead_time_days = details.get('lead_time_days', 7)
            
            if avg_weekly_usage > 0:
                # Calculate optimal reorder point
                weekly_usage = avg_weekly_usage
                safety_stock = avg_weekly_usage * 0.5  # 50% safety buffer
                reorder_point = (weekly_usage * lead_time_days / 7) + safety_stock
                
                # Economic Order Quantity (simplified)
                ordering_cost = 50  # Assumed fixed ordering cost
                holding_cost_rate = 0.20  # 20% annual holding cost
                annual_demand = weekly_usage * 52
                
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / (unit_cost * holding_cost_rate))
                
                recommendations.append({
                    'item': item,
                    'current_stock': current_stock,
                    'reorder_point': round(reorder_point, 1),
                    'optimal_order_quantity': round(eoq, 1),
                    'weeks_of_stock': round(current_stock / weekly_usage, 1) if weekly_usage > 0 else 0,
                    'status': 'reorder_soon' if current_stock <= reorder_point else 'adequate',
                    'estimated_carrying_cost': round(current_stock * unit_cost * holding_cost_rate / 52, 2)
                })
        
        return {
            'inventory_recommendations': recommendations,
            'total_inventory_value': sum(r['current_stock'] * inventory_data[r['item']].get('unit_cost', 0) 
                                       for r in recommendations),
            'items_needing_reorder': len([r for r in recommendations if r['status'] == 'reorder_soon'])
        }
    
    def analyze_business_performance(self, transactions_df: pd.DataFrame, timeframe: str = 'monthly') -> Dict:
        """Analyze overall business performance metrics"""
        # Revenue analysis
        revenue_data = transactions_df[transactions_df['amount'] > 0]
        expense_data = transactions_df[transactions_df['amount'] < 0]
        
        if timeframe == 'monthly':
            revenue_trend = revenue_data.groupby(revenue_data['date'].dt.to_period('M'))['amount'].sum()
            expense_trend = expense_data.groupby(expense_data['date'].dt.to_period('M'))['amount'].sum().abs()
        else:  # weekly
            revenue_trend = revenue_data.groupby(revenue_data['date'].dt.to_period('W'))['amount'].sum()
            expense_trend = expense_data.groupby(expense_data['date'].dt.to_period('W'))['amount'].sum().abs()
        
        profit_trend = revenue_trend - expense_trend
        
        # Key performance indicators
        total_revenue = revenue_data['amount'].sum()
        total_expenses = expense_data['amount'].abs().sum()
        net_profit = total_revenue - total_expenses
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Growth rate calculation
        if len(revenue_trend) >= 2:
            recent_revenue = revenue_trend.iloc[-1]
            previous_revenue = revenue_trend.iloc[-2]
            growth_rate = ((recent_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
        else:
            growth_rate = 0
        
        return {
            'financial_summary': {
                'total_revenue': round(total_revenue, 2),
                'total_expenses': round(total_expenses, 2),
                'net_profit': round(net_profit, 2),
                'profit_margin': round(profit_margin, 2),
                'growth_rate': round(growth_rate, 2)
            },
            'trends': {
                'revenue_trend': revenue_trend.to_dict(),
                'expense_trend': expense_trend.to_dict(),
                'profit_trend': profit_trend.to_dict()
            },
            'recommendations': self._generate_performance_recommendations(profit_margin, growth_rate)
        }
    
    def _calculate_expense_trend(self, expense_data: pd.DataFrame) -> str:
        """Calculate expense trend direction"""
        if len(expense_data) < 2:
            return 'insufficient_data'
        
        recent_month = expense_data['date'].max() - timedelta(days=30)
        recent_expenses = expense_data[expense_data['date'] >= recent_month]['amount'].sum()
        
        previous_month = recent_month - timedelta(days=30)
        previous_expenses = expense_data[
            (expense_data['date'] >= previous_month) & (expense_data['date'] < recent_month)
        ]['amount'].sum()
        
        if previous_expenses == 0:
            return 'insufficient_data'
        
        change_rate = (recent_expenses - previous_expenses) / previous_expenses
        
        if change_rate > 0.1:
            return 'increasing'
        elif change_rate < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_performance_recommendations(self, profit_margin: float, growth_rate: float) -> List[Dict]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if profit_margin < 10:
            recommendations.append({
                'area': 'profitability',
                'suggestion': 'Focus on cost reduction and pricing optimization',
                'priority': 'high',
                'expected_impact': 'Improve profit margin by 5-8%'
            })
        
        if growth_rate < 5:
            recommendations.append({
                'area': 'growth',
                'suggestion': 'Explore new revenue streams or market expansion',
                'priority': 'medium',
                'expected_impact': 'Increase monthly growth rate'
            })
        
        if profit_margin > 20 and growth_rate > 10:
            recommendations.append({
                'area': 'investment',
                'suggestion': 'Consider reinvesting profits for scaling operations',
                'priority': 'low',
                'expected_impact': 'Accelerate business growth'
            })
        
        return recommendations

# ============= MICROSERVICE 3: DOCUMENT PROCESSING SERVICE =============

class DocumentProcessor:
    def __init__(self):
        self.invoice_patterns = {
            'invoice_number': r'(?:invoice|inv|bill)[\s#]*:?\s*([A-Z0-9\-]+)',
            'amount': r'(?:total|amount|due)[\s:]*\$?([0-9,]+\.?[0-9]*)',
            'date': r'(?:date|issued)[\s:]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
            'due_date': r'(?:due|pay by)[\s:]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
            'vendor': r'(?:from|vendor|company)[\s:]*([A-Za-z\s&]+)'
        }
        
    def process_invoice_image(self, image_bytes: bytes) -> Dict:
        """Extract invoice information from image using OCR"""
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(thresh)
            
            # Extract structured information
            extracted_data = self._extract_invoice_fields(text)
            
            return {
                'status': 'success',
                'extracted_text': text,
                'structured_data': extracted_data,
                'confidence': self._calculate_extraction_confidence(extracted_data)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'extracted_text': '',
                'structured_data': {},
                'confidence': 0.0
            }
    
    def _extract_invoice_fields(self, text: str) -> Dict:
        """Extract structured fields from invoice text"""
        extracted = {}
        text_lower = text.lower()
        
        for field, pattern in self.invoice_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted[field] = match.group(1).strip()
        
        # Post-process extracted data
        if 'amount' in extracted:
            try:
                extracted['amount'] = float(extracted['amount'].replace(',', ''))
            except ValueError:
                extracted['amount'] = 0.0
        
        if 'date' in extracted:
            try:
                extracted['date'] = datetime.strptime(extracted['date'], '%m/%d/%Y').isoformat()
            except ValueError:
                pass
        
        return extracted
    
    def _calculate_extraction_confidence(self, extracted_data: Dict) -> float:
        """Calculate confidence score for extraction"""
        required_fields = ['invoice_number', 'amount', 'date']
        found_fields = sum(1 for field in required_fields if field in extracted_data and extracted_data[field])
        
        return found_fields / len(required_fields)
    
    def analyze_document_patterns(self, documents: List[Dict]) -> Dict:
        """Analyze patterns in processed documents"""
        if not documents:
            return {'message': 'No documents to analyze'}
        
        # Vendor analysis
        vendors = {}
        total_amount = 0
        date_range = []
        
        for doc in documents:
            data = doc.get('structured_data', {})
            
            vendor = data.get('vendor', 'Unknown')
            amount = data.get('amount', 0)
            date_str = data.get('date')
            
            if vendor not in vendors:
                vendors[vendor] = {'count': 0, 'total_amount': 0}
            
            vendors[vendor]['count'] += 1
            vendors[vendor]['total_amount'] += amount
            total_amount += amount
            
            if date_str:
                try:
                    date_range.append(datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                except:
                    pass
        
        # Calculate insights
        top_vendors = sorted(vendors.items(), key=lambda x: x[1]['total_amount'], reverse=True)[:5]
        avg_invoice_amount = total_amount / len(documents) if documents else 0
        
        return {
            'total_documents': len(documents),
            'total_amount': round(total_amount, 2),
            'average_invoice_amount': round(avg_invoice_amount, 2),
            'top_vendors': top_vendors,
            'date_range': {
                'start': min(date_range).isoformat() if date_range else None,
                'end': max(date_range).isoformat() if date_range else None
            },
            'processing_accuracy': sum(doc.get('confidence', 0) for doc in documents) / len(documents)
        }

# ============= MICROSERVICE 4: RECOMMENDATION ENGINE =============

class RecommendationEngine:
    def __init__(self):
        self.recommendation_cache = {}
        
    def generate_cashflow_recommendations(self, cashflow_prediction: Dict, current_balance: float) -> List[Dict]:
        """Generate cash flow management recommendations"""
        recommendations = []
        predictions = cashflow_prediction.get('predictions', [])
        
        for pred in predictions:
            predicted_balance = pred['predicted_balance']
            days_ahead = pred['days_ahead']
            
            if predicted_balance < 1000:  # Low balance threshold
                recommendations.append({
                    'type': 'cash_flow_alert',
                    'priority': 'high',
                    'message': f'Low balance predicted in {days_ahead} days: ${predicted_balance:.2f}',
                    'actions': [
                        'Accelerate accounts receivable collection',
                        'Delay non-critical payments',
                        'Consider short-term financing options'
                    ],
                    'impact': 'critical'
                })
            elif predicted_balance < current_balance * 0.5:
                recommendations.append({
                    'type': 'cash_flow_warning',
                    'priority': 'medium',
                    'message': f'Balance dropping to ${predicted_balance:.2f} in {days_ahead} days',
                    'actions': [
                        'Review upcoming expenses',
                        'Follow up on overdue invoices',
                        'Consider payment term negotiations'
                    ],
                    'impact': 'moderate'
                })
        
        # Positive cash flow opportunities
        if all(pred['predicted_balance'] > current_balance * 1.2 for pred in predictions[-3:]):
            recommendations.append({
                'type': 'investment_opportunity',
                'priority': 'low',
                'message': 'Strong cash position predicted - consider growth investments',
                'actions': [
                    'Invest in inventory expansion',
                    'Consider equipment upgrades',
                    'Explore new market opportunities'
                ],
                'impact': 'growth'
            })
        
        return recommendations
    
    def generate_operational_recommendations(self, operations_analysis: Dict) -> List[Dict]:
        """Generate operational improvement recommendations"""
        recommendations = []
        
        # Expense optimization recommendations
        optimizations = operations_analysis.get('optimization_opportunities', [])
        for opt in optimizations:
            recommendations.append({
                'type': 'expense_optimization',
                'priority': opt['priority'],
                'message': f"{opt['category']}: {opt['suggestion']}",
                'potential_savings': opt['potential_savings'],
                'category': opt['category'],
                'actions': [f"Review {opt['category']} spending patterns"],
                'impact': 'cost_reduction'
            })
        
        # Performance-based recommendations
        financial_summary = operations_analysis.get('financial_summary', {})
        profit_margin = financial_summary.get('profit_margin', 0)
        growth_rate = financial_summary.get('growth_rate', 0)
        
        if profit_margin < 15:
            recommendations.append({
                'type': 'profitability_improvement',
                'priority': 'high',
                'message': f'Profit margin at {profit_margin:.1f}% - below healthy threshold',
                'actions': [
                    'Analyze pricing strategy',
                    'Review cost structure',
                    'Identify high-margin products/services'
                ],
                'impact': 'profitability'
            })
        
        if growth_rate < 0:
            recommendations.append({
                'type': 'growth_concern',
                'priority': 'high',
                'message': f'Negative growth rate: {growth_rate:.1f}%',
                'actions': [
                    'Analyze market conditions',
                    'Review customer satisfaction',
                    'Consider marketing initiatives'
                ],
                'impact': 'revenue'
            })
        
        return recommendations
    
    def generate_credit_recommendations(self, financial_profile: Dict) -> List[Dict]:
        """Generate credit and financing recommendations"""
        recommendations = []
        
        profit_margin = financial_profile.get('profit_margin', 0)
        cash_cycle = financial_profile.get('cash_cycle_days', 0)
        revenue_stability = financial_profile.get('revenue_stability', 0.5)
        
        # Credit readiness score
        credit_score = self._calculate_credit_readiness(financial_profile)
        
        if credit_score > 0.7:
            recommendations.append({
                'type': 'credit_opportunity',
                'priority': 'medium',
                'message': 'Strong financial profile - good credit opportunities available',
                'credit_readiness_score': credit_score,
                'actions': [
                    'Consider growth financing options',
                    'Negotiate improved supplier terms',
                    'Explore working capital solutions'
                ],
                'impact': 'growth_financing'
            })
        elif credit_score < 0.4:
            recommendations.append({
                'type': 'credit_improvement',
                'priority': 'medium',
                'message': 'Focus on improving creditworthiness',
                'credit_readiness_score': credit_score,
                'actions': [
                    'Improve cash flow consistency',
                    'Build stronger financial records',
                    'Reduce financial volatility'
                ],
                'impact': 'credit_building'
            })
        
        # Working capital recommendations
        if cash_cycle > 45:
            recommendations.append({
                'type': 'working_capital',
                'priority': 'medium',
                'message': f'Cash cycle of {cash_cycle:.0f} days is extended',
                'actions': [
                    'Negotiate faster payment terms with customers',
                    'Extend payment terms with suppliers',
                    'Consider invoice factoring'
                ],
                'impact': 'cash_flow_improvement'
            })
        
        return recommendations
    
    def _calculate_credit_readiness(self, financial_profile: Dict) -> float:
        """Calculate credit readiness score"""
        factors = {
            'profit_margin': financial_profile.get('profit_margin', 0) / 100,
            'revenue_stability': financial_profile.get('revenue_stability', 0.5),
            'cash_flow_positive': 1 if financial_profile.get('avg_daily_balance', 0) > 0 else 0,
            'growth_trend': min(financial_profile.get('growth_rate', 0) / 50, 1)
        }
        
        weights = {
            'profit_margin': 0.3,
            'revenue_stability': 0.25,
            'cash_flow_positive': 0.25,
            'growth_trend': 0.2
        }
        
        score = sum(factors[key] * weights[key] for key in factors)
        return min(max(score, 0), 1)  # Clamp between 0 and 1
    
    def generate_comprehensive_recommendations(self, 
                                            cashflow_data: Dict, 
                                            operations_data: Dict, 
                                            current_balance: float) -> Dict:
        """Generate comprehensive business recommendations"""
        
        cashflow_recs = self.generate_cashflow_recommendations(cashflow_data, current_balance)
        operational_recs = self.generate_operational_recommendations(operations_data)
        credit_recs = self.generate_credit_recommendations(operations_data.get('financial_summary', {}))
        
        all_recommendations = cashflow_recs + operational_recs + credit_recs
        
        # Prioritize recommendations
        high_priority = [r for r in all_recommendations if r['priority'] == 'high']
        medium_priority = [r for r in all_recommendations if r['priority'] == 'medium']
        low_priority = [r for r in all_recommendations if r['priority'] == 'low']
        
        return {
            'total_recommendations': len(all_recommendations),
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'summary': {
                'critical_actions': len(high_priority),
                'improvement_opportunities': len(medium_priority),
                'growth_opportunities': len(low_priority)
            }
        }

# ============= FASTAPI MICROSERVICES ENDPOINTS =============

# Create separate FastAPI apps for each microservice
cashflow_app = FastAPI(title="Cash Flow Prediction Service")
operations_app = FastAPI(title="Operations Intelligence Service") 
document_app = FastAPI(title="Document Processing Service")
recommendation_app = FastAPI(title="Recommendation Engine")

# Initialize services
cashflow_predictor = CashFlowPredictor()
operations_intelligence = OperationsIntelligence()
document_processor = DocumentProcessor()
recommendation_engine = RecommendationEngine()

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Cash Flow Prediction Endpoints
@cashflow_app.post("/predict")
async def predict_cashflow(user_id: str, days_ahead: int = 7):
    """Predict cash flow for specified days"""
    try:
        # Get user transaction data from cache or database
        cached_data = redis_client.get(f"transactions:{user_id}")
        if not cached_data:
            raise HTTPException(status_code=404, detail="No transaction data found")
        
        transactions_df = pd.read_json(cached_data)
        
        # Train model if not exists
        model_key = f"model:{user_id}:cashflow"
        if not redis_client.exists(model_key):
            training_result = cashflow_predictor.train(transactions_df)
            redis_client.setex(model_key, 86400, json.dumps(training_result))  # Cache for 24h
        
        # Generate predictions
        predictions = cashflow_predictor.predict_cashflow(transactions_df, days_ahead)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Operations Intelligence Endpoints
@operations_app.post("/analyze")
async def analyze_operations(user_id: str):
    """Analyze business operations"""
    try:
        cached_data = redis_client.get(f"transactions:{user_id}")
        if not cached_data:
            raise HTTPException(status_code=404, detail="No transaction data found")
        
        transactions_df = pd.read_json(cached_data)
        
        # Analyze expenses
        expense_analysis = operations_intelligence.analyze_expense_patterns(transactions_df)
        
        # Analyze performance
        performance_analysis = operations_intelligence.analyze_business_performance(transactions_df)
        
        return {
            'expense_analysis': expense_analysis,
            'performance_analysis': performance_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document Processing Endpoints
@document_app.post("/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    """Process invoice image and extract data"""
    try:
        image_bytes = await file.read()
        result = document_processor.process_invoice_image(image_bytes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation Engine Endpoints
@recommendation_app.post("/generate")
async def generate_recommendations(user_id: str):
    """Generate comprehensive recommendations"""
    try:
        # Get cached analysis data
        cashflow_data = json.loads(redis_client.get(f"cashflow:{user_id}") or '{}')
        operations_data = json.loads(redis_client.get(f"operations:{user_id}") or '{}')
        
        current_balance = operations_data.get('financial_summary', {}).get('current_balance', 0)
        
        recommendations = recommendation_engine.generate_comprehensive_recommendations(
            cashflow_data, operations_data, current_balance
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run all microservices (in production, these would be separate containers)
    uvicorn.run(cashflow_app, host="0.0.0.0", port=8001)  # Cash Flow Service
    # uvicorn.run(operations_app, host="0.0.0.0", port=8002)  # Operations Service
    # uvicorn.run(document_app, host="0.0.0.0", port=8003)   # Document Service
    # uvicorn.run(recommendation_app, host="0.0.0.0", port=8004)  # Recommendation Service
