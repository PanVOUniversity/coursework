import React from 'react';
import { motion } from 'framer-motion';
import { FaCheck } from 'react-icons/fa';
import './styles.css';

const Pricing = () => {
  const plans = [
    { name: 'Базовый', price: '2000', features: ['Стрижка', 'Укладка', 'Консультация'] },
    { name: 'Стандарт', price: '4000', features: ['Стрижка', 'Окрашивание', 'Укладка', 'Уход'], popular: true },
    { name: 'Премиум', price: '8000', features: ['Стрижка', 'Окрашивание', 'Укладка', 'Процедуры', 'Консультация'] }
  ];

  return (
    <section id="pricing" className="pricing">
      <div className="container">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="section-title">Цены</h2>
        </motion.div>

        <div className="pricing-grid">
          {plans.map((plan, index) => (
            <motion.div key={index} className={`pricing-card ${plan.popular ? 'popular' : ''}`} initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} 
              transition={{ delay: index * 0.2 }} viewport={{ once: true }} whileHover={{ y: -10 }}>
              {plan.popular && <div className="popular-badge">Популярное</div>}
              <h3 className="plan-name">{plan.name}</h3>
              <div className="plan-price">₽{plan.price}</div>
              <ul className="plan-features">
                {plan.features.map((feature, i) => (
                  <li key={i}><FaCheck /> {feature}</li>
                ))}
              </ul>
              <motion.button className="btn btn-primary" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>Записаться</motion.button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Pricing;
